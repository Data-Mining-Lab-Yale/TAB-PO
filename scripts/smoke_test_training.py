#!/usr/bin/env python3
"""Offline CPU SFT -> reference -> TAB-PO -> adapter reload on a tiny model.

This is a functional check, not a SciERC experiment. No model is downloaded.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def run():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, PeftModel
    from train_sft import build_features, SimpleCollator
    from train_tabpo import Config, DPODataCollatorTokenWeighted, TWCDPOTrainer, save_lora_adapter_only_zero3, add_reference_logps_to_dataset
    set_seed(42)
    torch.set_num_threads(1)
    chosen = {"entities": [{"entity_id": "E1", "text": "parser", "type": "Method", "sentence_id": 0}],
              "relations": [], "coreference_clusters": []}
    rejected = copy.deepcopy(chosen)
    rejected["entities"][0]["type"] = "Task"
    chosen, rejected = (json.dumps(x, separators=(",", ":")) for x in (chosen, rejected))
    query = "Extract scientific entities from this text: parser."
    special = ["[UNK]", "[PAD]", "[END]", "[USER]", "[ASSISTANT]"]
    words = sorted(set(query.split()) | {chosen, rejected})
    vocab = {word: i for i, word in enumerate(special + [w for w in words if w not in special])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[END]")
    tok.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}[USER] {{ message['content'] }} {% elif message['role'] == 'assistant' %}[ASSISTANT] {{ message['content'] }} [END]{% endif %}{% endfor %}{% if add_generation_prompt %}[ASSISTANT] {% endif %}"
    base = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=128, pad_token_id=tok.pad_token_id,
                                      bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id))
    sft_feature = build_features({"query": query, "answer": chosen}, tok, 128)
    if not any(x == -100 for x in sft_feature["labels"]) or not any(x != -100 for x in sft_feature["labels"]):
        raise RuntimeError("Completion masking is invalid")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        common = dict(use_cpu=True, report_to="none", max_steps=2, per_device_train_batch_size=1,
                      save_strategy="no", logging_steps=1, disable_tqdm=True, learning_rate=1e-3)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        sft = Trainer(model=model, args=TrainingArguments(output_dir=str(tmp / "sft"), **common),
                      train_dataset=[sft_feature], data_collator=SimpleCollator(tokenizer=tok))
        sft.train()
        model = model.merge_and_unload()
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")
        model.config.use_cache = False
        cfg = Config(model_name="tiny-local", train_data_path="unused", valid_data_path="unused", output_dir=str(tmp / "adapter"),
                     max_length=128, max_prompt_length=64, bf16=False, gradient_checkpointing=False,
                     lora_r=2, lora_alpha=4, lora_dropout=0)
        collator = DPODataCollatorTokenWeighted(tok, cfg, system_prompt="")
        pair = {"prompt": query, "chosen": chosen, "rejected": rejected}
        batch = collator([pair])
        class MemoryPairs(list):
            @property
            def column_names(self):
                return list(self[0])

            def add_column(self, name, values):
                return MemoryPairs(dict(row, **{name: value}) for row, value in zip(self, values))

        cached = add_reference_logps_to_dataset(MemoryPairs([pair]), model, collator, cfg, "smoke")
        # SFT and TAB-PO must tokenize/mask the identical preferred chat.
        if batch["chosen_input_ids"][0].tolist() != sft_feature["input_ids"] or batch["chosen_labels"][0].tolist() != sft_feature["labels"]:
            raise RuntimeError("SFT and preference formatting diverge")
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        trainer = TWCDPOTrainer(model=model, cfg=cfg,
                               args=TrainingArguments(output_dir=str(tmp / "tabpo"), remove_unused_columns=False, **common),
                               train_dataset=cached, data_collator=collator)
        model.eval()
        direct_loss = trainer.compute_loss(model, collator([pair])).detach()
        cached_loss = trainer.compute_loss(model, collator(cached)).detach()
        if not torch.allclose(direct_loss, cached_loss, atol=1e-6):
            raise RuntimeError("Cached reference loss differs from the adapter-off reference loss")
        result = trainer.train()
        if not torch.isfinite(torch.tensor(result.training_loss)):
            raise RuntimeError("TAB-PO loss is not finite")
        save_lora_adapter_only_zero3(trainer, tok, str(tmp / "adapter"))
        model.eval()
        inp = {"input_ids": batch["chosen_input_ids"], "attention_mask": batch["chosen_attention_mask"]}
        with torch.no_grad():
            expected = model(**inp).logits.clone()
        trained_base = model.unload()
        if hasattr(trained_base, "peft_config"):
            delattr(trained_base, "peft_config")
        reloaded = PeftModel.from_pretrained(trained_base, str(tmp / "adapter")).eval()
        with torch.no_grad():
            actual = reloaded(**inp).logits
            with reloaded.disable_adapter():
                baseline = reloaded(**inp).logits
        if not torch.allclose(expected, actual, atol=1e-6):
            raise RuntimeError("Saved adapter does not reproduce trained logits")
        if torch.allclose(actual, baseline, atol=1e-7):
            raise RuntimeError("Reloaded adapter has no functional effect")
        return {"sft_steps": 2, "tabpo_steps": 2, "finite_loss": True,
                "prompt_format_consistent": True, "adapter_reload_verified": True,
                "reference_cache_verified": True, "model_downloaded": False, "device": "cpu"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
