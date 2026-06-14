#!/usr/bin/env python3
"""Manual semantic extraction for graphify chunks that failed via subagents."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

def sanitize(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"

def node_id(parent_dir: str, filename: str, entity: str) -> str:
    stem = f"{sanitize(parent_dir)}_{sanitize(filename)}"
    return f"{stem}_{sanitize(entity)}"

def extract_py_semantic(path: Path) -> tuple[list[dict], list[dict]]:
    """Extract semantic nodes and edges from a Python file."""
    nodes: list[dict] = []
    edges: list[dict] = []
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return nodes, edges

    parent = path.parent.name or "scripts"
    filename = path.stem
    rel = str(path.relative_to(Path(".")))

    # Module-level docstring as rationale node
    doc = ast.get_docstring(tree)
    if doc:
        nodes.append({
            "id": node_id(parent, filename, "module_docstring"),
            "label": f"{filename} module docstring",
            "file_type": "rationale",
            "source_file": rel,
            "source_location": None,
            "source_url": None,
            "captured_at": None,
            "author": None,
            "contributor": None,
        })

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            nid = node_id(parent, filename, node.name)
            nodes.append({
                "id": nid,
                "label": node.name,
                "file_type": "code",
                "source_file": rel,
                "source_location": node.lineno,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None,
            })
            # Inheritance edges
            for base in node.bases:
                if isinstance(base, ast.Name):
                    edges.append({
                        "source": nid,
                        "target": node_id(parent, filename, base.id),
                        "relation": "implements",
                        "confidence": "EXTRACTED",
                        "confidence_score": 1.0,
                        "source_file": rel,
                        "source_location": node.lineno,
                        "weight": 1.0,
                    })
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            nid = node_id(parent, filename, node.name)
            nodes.append({
                "id": nid,
                "label": f"{node.name}()",
                "file_type": "code",
                "source_file": rel,
                "source_location": node.lineno,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None,
            })
            # calls edges from function body
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        edges.append({
                            "source": nid,
                            "target": node_id(parent, filename, child.func.id),
                            "relation": "calls",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": rel,
                            "source_location": child.lineno if hasattr(child, 'lineno') else None,
                            "weight": 1.0,
                        })
                    elif isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                        edges.append({
                            "source": nid,
                            "target": node_id(parent, filename, child.func.attr),
                            "relation": "calls",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": rel,
                            "source_location": child.lineno if hasattr(child, 'lineno') else None,
                            "weight": 1.0,
                        })

    # String-reference edges: find module names mentioned in strings
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            # Cross-module references via string literals
            for mod in ["cascade_runner", "deal_finder", "paper_trading", "photo_arbitrage",
                        "full_scrape_model", "benchmark_basic_to_full", "teacher_student",
                        "time_to_sell", "telegram_implementation", "analysis_pipeline"]:
                if mod in s.lower():
                    edges.append({
                        "source": node_id(parent, filename, "module"),
                        "target": node_id(mod.replace("_", ""), mod, "module"),
                        "relation": "references",
                        "confidence": "INFERRED",
                        "confidence_score": 0.65,
                        "source_file": rel,
                        "source_location": node.lineno if hasattr(node, 'lineno') else None,
                        "weight": 1.0,
                    })

    return nodes, edges

def extract_doc_md(path: Path) -> list[dict]:
    """Extract concept nodes from markdown docs."""
    nodes: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return nodes
    rel = str(path.relative_to(Path(".")))
    parent = path.parent.name or "docs"
    filename = path.stem

    # Heading-based concept nodes
    for match in re.finditer(r"^##+\s+(.+)$", text, re.MULTILINE):
        title = match.group(1).strip()
        nid = node_id(parent, filename, title)
        nodes.append({
            "id": nid,
            "label": title,
            "file_type": "document",
            "source_file": rel,
            "source_location": None,
            "source_url": None,
            "captured_at": None,
            "author": None,
            "contributor": None,
        })
    return nodes

def process_chunk(file_list_path: Path, out_path: Path) -> dict[str, Any]:
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    root = Path(".").resolve()

    for line in file_list_path.read_text(encoding="utf-8").strip().split("\n"):
        fpath = Path(line.strip())
        if not fpath.exists():
            continue
        if fpath.suffix == ".py":
            n, e = extract_py_semantic(fpath)
            all_nodes.extend(n)
            all_edges.extend(e)
        elif fpath.suffix == ".md":
            all_nodes.extend(extract_doc_md(fpath))

    # Deduplicate nodes by id
    seen = set()
    deduped = []
    for n in all_nodes:
        if n["id"] not in seen:
            seen.add(n["id"])
            deduped.append(n)

    result = {
        "nodes": deduped,
        "edges": all_edges,
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}: {len(deduped)} nodes, {len(all_edges)} edges")
    return result

if __name__ == "__main__":
    import sys
    for i, list_path in enumerate(sys.argv[1:-1], start=0):
        out = Path(sys.argv[-1]).parent / f".graphify_chunk_{i:02d}_manual.json"
        process_chunk(Path(list_path), out)
