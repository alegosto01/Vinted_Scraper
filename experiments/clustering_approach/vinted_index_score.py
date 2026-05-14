#!/usr/bin/env python3
"""
vinted_index_store.py

SQLite-backed store for:
- product index (centroids, thresholds, core token counts)
- variant index (centroids, thresholds, rolling price buffers)
- processed pointers (per CSV file)

Install:
  python -m pip install numpy pandas
"""

import os
import json
import sqlite3
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
  product_id INTEGER PRIMARY KEY,
  centroid BLOB NOT NULL,
  dim INTEGER NOT NULL,
  n INTEGER NOT NULL,
  product_threshold REAL NOT NULL,
  canonical_name TEXT,
  block_key_hint TEXT,
  core_token_counts TEXT,         -- json dict token->count
  core_token_min_frac REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS variants (
  variant_id INTEGER PRIMARY KEY,
  product_id INTEGER NOT NULL,
  centroid BLOB NOT NULL,
  dim INTEGER NOT NULL,
  n INTEGER NOT NULL,
  variant_threshold REAL NOT NULL,
  price_weight REAL NOT NULL,
  core_frac REAL NOT NULL,
  price_buffer TEXT,              -- json list of floats (rolling)
  variant_text_top TEXT,
  FOREIGN KEY(product_id) REFERENCES products(product_id)
);

CREATE INDEX IF NOT EXISTS idx_variants_product ON variants(product_id);

CREATE TABLE IF NOT EXISTS processed_files (
  filepath TEXT PRIMARY KEY,
  last_row_hash TEXT,
  last_seen_rows INTEGER DEFAULT 0
);
"""

def _to_blob(vec: np.ndarray) -> bytes:
    vec = np.asarray(vec, dtype=np.float32)
    return vec.tobytes()

def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)

def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.executescript(SCHEMA)
    return con

def set_meta(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, json.dumps(value)))
    con.commit()

def get_meta(con: sqlite3.Connection, key: str, default=None):
    cur = con.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else default

def upsert_product(con: sqlite3.Connection, product_id: int, centroid: np.ndarray, n: int,
                   product_threshold: float, canonical_name: str,
                   block_key_hint: str,
                   core_token_counts: Dict[str, int],
                   core_token_min_frac: float) -> None:
    blob = _to_blob(centroid)
    dim = int(np.asarray(centroid).shape[0])
    con.execute("""
      INSERT OR REPLACE INTO products
      (product_id, centroid, dim, n, product_threshold, canonical_name, block_key_hint, core_token_counts, core_token_min_frac)
      VALUES (?,?,?,?,?,?,?,?,?)
    """, (product_id, blob, dim, n, float(product_threshold), canonical_name, block_key_hint, json.dumps(core_token_counts), float(core_token_min_frac)))
    con.commit()

def upsert_variant(con: sqlite3.Connection, variant_id: int, product_id: int, centroid: np.ndarray, n: int,
                   variant_threshold: float, price_weight: float, core_frac: float,
                   price_buffer: List[float], variant_text_top: str) -> None:
    blob = _to_blob(centroid)
    dim = int(np.asarray(centroid).shape[0])
    con.execute("""
      INSERT OR REPLACE INTO variants
      (variant_id, product_id, centroid, dim, n, variant_threshold, price_weight, core_frac, price_buffer, variant_text_top)
      VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (variant_id, product_id, blob, dim, n,
          float(variant_threshold), float(price_weight), float(core_frac),
          json.dumps(price_buffer), variant_text_top))
    con.commit()

def load_products(con: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    out = {}
    cur = con.execute("""
      SELECT product_id, centroid, dim, n, product_threshold, canonical_name, block_key_hint, core_token_counts, core_token_min_frac
      FROM products
    """)
    for (pid, blob, dim, n, thr, cname, bkh, ctc, minf) in cur.fetchall():
        out[int(pid)] = {
            "product_id": int(pid),
            "centroid": _from_blob(blob, dim),
            "dim": int(dim),
            "n": int(n),
            "product_threshold": float(thr),
            "canonical_name": cname or "",
            "block_key_hint": bkh or "",
            "core_token_counts": json.loads(ctc) if ctc else {},
            "core_token_min_frac": float(minf),
        }
    return out

def load_variants(con: sqlite3.Connection) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    returns {product_id: {variant_id: variant_info}}
    """
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    cur = con.execute("""
      SELECT variant_id, product_id, centroid, dim, n, variant_threshold, price_weight, core_frac, price_buffer, variant_text_top
      FROM variants
    """)
    for (vid, pid, blob, dim, n, vthr, pwt, cfrac, pb, vtt) in cur.fetchall():
        pid = int(pid); vid = int(vid)
        out.setdefault(pid, {})
        out[pid][vid] = {
            "variant_id": vid,
            "product_id": pid,
            "centroid": _from_blob(blob, dim),
            "dim": int(dim),
            "n": int(n),
            "variant_threshold": float(vthr),
            "price_weight": float(pwt),
            "core_frac": float(cfrac),
            "price_buffer": json.loads(pb) if pb else [],
            "variant_text_top": vtt or "",
        }
    return out

def get_next_ids(con: sqlite3.Connection) -> Tuple[int, int]:
    """
    returns (next_product_id, next_variant_id)
    """
    cur = con.execute("SELECT COALESCE(MAX(product_id), -1) FROM products")
    maxp = cur.fetchone()[0]
    cur = con.execute("SELECT COALESCE(MAX(variant_id), -1) FROM variants")
    maxv = cur.fetchone()[0]
    return int(maxp) + 1, int(maxv) + 1

def update_processed_state(con: sqlite3.Connection, filepath: str, last_seen_rows: int) -> None:
    con.execute("""
      INSERT OR REPLACE INTO processed_files(filepath, last_seen_rows, last_row_hash)
      VALUES(?, ?, COALESCE((SELECT last_row_hash FROM processed_files WHERE filepath=?), ''))
    """, (filepath, int(last_seen_rows), filepath))
    con.commit()

def get_processed_state(con: sqlite3.Connection, filepath: str) -> int:
    cur = con.execute("SELECT last_seen_rows FROM processed_files WHERE filepath=?", (filepath,))
    row = cur.fetchone()
    return int(row[0]) if row else 0
