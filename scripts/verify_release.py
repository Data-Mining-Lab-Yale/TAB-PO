#!/usr/bin/env python3
"""Check public release contents, local Markdown links, credentials, and syntax."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    errors = []
    paths = [p for p in root.rglob("*") if p.is_file() and
             not any(part in {".git", "__pycache__", ".venv", "outputs", ".verification", ".pytest_cache"}
                     for part in p.relative_to(root).parts)]
    expected_paper = {"README.md", "TAB_PO__main_paper.pdf", "TAB_PO__supplementary_material.pdf"}
    if {p.name for p in (root / "paper").iterdir()} != expected_paper:
        errors.append("paper/ must contain only README.md, TAB_PO__main_paper.pdf, and TAB_PO__supplementary_material.pdf")
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in {".zip", ".tex", ".bib", ".ipynb"}:
            errors.append(f"Private source/archive included: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        if path.suffix.lower() not in {".py", ".md", ".sh", ".txt", ".yml", ".yaml", ".toml", ".cff", ".json"}:
            continue
        content = path.read_text(encoding="utf-8")
        if re.search(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", content):
            errors.append(f"Potential credential in {relative}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", content):
                target = target.split("#", 1)[0].strip("<>")
                if not target or "://" in target or target.startswith(("mailto:", "app:")):
                    continue
                if not (path.parent / target).exists():
                    errors.append(f"Broken local link in {relative}: {target}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"files_checked": len(paths), "paper_pdf_only": True, "markdown_links_valid": True,
            "credential_patterns_found": 0, "python_syntax_valid": True}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
