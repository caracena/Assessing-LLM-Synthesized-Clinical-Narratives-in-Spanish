"""Normalize heterogeneous clinical seed collections into a standard JSON schema.

Converts Hugging Face datasets, CSV/TSV, JSON/JSONL, plain TXT, or BRAT folders
into `{text, source, source_detail}` JSON artifacts with deterministic config hashing.
"""

import csv
import hashlib
import json
import re
from pathlib import Path
from tqdm import tqdm

SCHEMA_VERSION = 2


def _get_loaders():
    return {
        "huggingface": _load_huggingface,
        "csv": _load_csv,
        "tsv": _load_tsv,
        "json": _load_json,
        "jsonl": _load_jsonl,
        "txt": _load_txt,
        "brat": _load_brat,
    }


HASH_FIELDS_DEFAULTS = {
    "name": None,
    "loader": None,
    "repo": None,
    "file_path": None,
    "text_column": "text",
    "source_column": None,
    "include_sources": None,
    "hf_split": None,
    "hf_subset": None,
    "min_text_length": 10,
    "max_text_length": 50000,
    "deduplicate": True,
}


def build_hash_config(job: dict) -> dict:
    """Return canonical hash configuration dictionary for a normalization job."""
    return {field: job.get(field, default) for field, default in HASH_FIELDS_DEFAULTS.items()}


def compute_config_hash(job: dict) -> str:
    """Compute deterministic 16-char SHA256 hash for a normalization job + SCHEMA_VERSION."""
    relevant = {
        "schema_version": SCHEMA_VERSION,
        **build_hash_config(job),
    }

    def _normalize(obj):
        if isinstance(obj, (list, set, tuple)):
            return sorted(str(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in sorted(obj.items())}
        return obj

    normalized = _normalize(relevant)
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def is_normalized_valid(output_path: str, expected_hash: str) -> bool:
    """Return True if output_path exists, is valid JSON, and matches expected_hash."""
    p = Path(output_path)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("metadata", {}).get("config_hash") == expected_hash


def normalize_corpus(
    name: str,
    loader: str,
    output_path: str,
    text_column: str = "text",
    repo: str = None,
    file_path: str = None,
    hf_split: str = None,
    hf_subset: str = None,
    source_column: str = None,
    include_sources: list = None,
    min_text_length: int = 10,
    max_text_length: int = 50000,
    deduplicate: bool = True,
    force: bool = False,
    **loader_kwargs,
) -> dict:
    """Normalize a seed source corpus to `{text, source, source_detail}` JSON format."""
    config_hash = compute_config_hash({
        "name": name, "loader": loader, "text_column": text_column,
        "repo": repo, "file_path": file_path, "hf_split": hf_split,
        "hf_subset": hf_subset, "source_column": source_column,
        "include_sources": include_sources, "min_text_length": min_text_length,
        "max_text_length": max_text_length, "deduplicate": deduplicate,
    })

    if not force and is_normalized_valid(output_path, config_hash):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[OK] {name}: up-to-date ({existing['metadata']['n_normalized']} texts) -- skipped")
        return {
            "output_path": output_path,
            "skipped": True,
            "n_normalized": existing["metadata"]["n_normalized"],
            "config_hash": config_hash,
        }

    print(f"\n{'='*60}")
    print(f"NORMALIZING CORPUS: {name}")
    print(f"Loader: {loader}")
    print(f"{'='*60}\n")

    loaders = _get_loaders()
    if loader not in loaders:
        raise ValueError(f"Unregistered loader '{loader}'. Available: {list(loaders.keys())}")

    loader_fn = loaders[loader]
    raw_texts, raw_sources = loader_fn(
        repo=repo,
        file_path=file_path,
        text_column=text_column,
        source_column=source_column,
        hf_split=hf_split,
        hf_subset=hf_subset,
        **loader_kwargs,
    )

    print(f"Loaded raw texts (before source filter): {len(raw_texts)}")

    if include_sources is not None:
        include_set = set(include_sources)
        filtered = [(t, s) for t, s in zip(raw_texts, raw_sources) if s in include_set]
        raw_texts = [t for t, s in filtered]
        raw_sources = [s for t, s in filtered]
        print(f"Texts matching include_sources={sorted(include_set)}: {len(raw_texts)}")

        missing_src = include_set - set(raw_sources)
        if missing_src and not raw_texts:
            print(f"  [WARN] None of include_sources appeared in dataset: {sorted(missing_src)}")

    normalized = []
    stats = {
        "total_loaded": len(raw_texts),
        "dropped_empty": 0,
        "dropped_short": 0,
        "dropped_long": 0,
        "dropped_duplicates": 0,
    }

    seen = set()
    for text, source_label in tqdm(zip(raw_texts, raw_sources), desc="Normalizing", total=len(raw_texts)):
        text = _clean_text(text)

        if not text:
            stats["dropped_empty"] += 1
            continue
        if len(text) < min_text_length:
            stats["dropped_short"] += 1
            continue
        if len(text) > max_text_length:
            stats["dropped_long"] += 1
            continue

        if deduplicate:
            key = text.lower().strip()
            if key in seen:
                stats["dropped_duplicates"] += 1
                continue
            seen.add(key)

        normalized.append({
            "text": text,
            "source": name,
            "source_detail": source_label,
        })

    stats["total_normalized"] = len(normalized)

    output = {
        "metadata": {
            "name": name,
            "loader": loader,
            "config_hash": config_hash,
            "schema_version": SCHEMA_VERSION,
            "n_normalized": len(normalized),
            "min_text_length": min_text_length,
            "max_text_length": max_text_length,
            "deduplicated": deduplicate,
            "stats": stats,
        },
        "texts": normalized,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  Total loaded:       {stats['total_loaded']}")
    print(f"  Dropped empty:      {stats['dropped_empty']}")
    print(f"  Dropped short:      {stats['dropped_short']}")
    print(f"  Dropped long:       {stats['dropped_long']}")
    print(f"  Dropped duplicates: {stats['dropped_duplicates']}")
    print(f"  Total normalized:   {stats['total_normalized']}")
    print(f"\n[OK] Saved to: {output_path}")

    return {
        "output_path": output_path,
        "skipped": False,
        "n_normalized": len(normalized),
        "config_hash": config_hash,
    }


def _load_huggingface(repo, text_column, source_column=None, hf_split=None, hf_subset=None, **_):
    from datasets import load_dataset

    print(f"Downloading from Hugging Face: {repo}...")
    kwargs = {}
    if hf_subset:
        kwargs["name"] = hf_subset

    ds = load_dataset(repo, **kwargs)

    split = hf_split
    if split is None:
        split = "train" if "train" in ds else list(ds.keys())[0]
        print(f"  Selected split: {split}")

    data = ds[split]

    if text_column not in data.column_names:
        raise ValueError(f"Column '{text_column}' not found. Available: {data.column_names}")

    texts = [str(row[text_column]) for row in data]
    if source_column:
        if source_column not in data.column_names:
            raise ValueError(f"source_column '{source_column}' not found. Available: {data.column_names}")
        sources = [str(row[source_column]) for row in data]
    else:
        sources = [None] * len(texts)

    return texts, sources


def _load_csv(file_path, text_column, source_column=None, **_):
    return _load_delimited(file_path, text_column, source_column, delimiter=",")


def _load_tsv(file_path, text_column, source_column=None, **_):
    return _load_delimited(file_path, text_column, source_column, delimiter="\t")


def _load_delimited(file_path, text_column, source_column, delimiter):
    texts, sources = [], []
    with open(file_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if text_column not in reader.fieldnames:
            raise ValueError(f"Column '{text_column}' not found. Available: {reader.fieldnames}")
        for row in reader:
            texts.append(row[text_column])
            sources.append(row.get(source_column) if source_column else None)
    return texts, sources


def _load_json(file_path, text_column, source_column=None, **_):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in [text_column, "texts", "data", "documents"]:
            if key in data and isinstance(data[key], list):
                rows = data[key]
                break

    if rows is None:
        raise ValueError(f"Unrecognized JSON format: {type(data)}")
    if len(rows) == 0:
        return [], []
    if isinstance(rows[0], str):
        return rows, [None] * len(rows)

    texts = [str(row.get(text_column, "")) for row in rows]
    sources = [
        str(row.get(source_column)) if source_column and row.get(source_column) is not None else None
        for row in rows
    ]
    return texts, sources


def _load_jsonl(file_path, text_column, source_column=None, **_):
    file_path = Path(file_path).expanduser()
    texts, sources = [], []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                texts.append(obj)
                sources.append(None)
            elif isinstance(obj, dict):
                texts.append(str(obj.get(text_column, "")))
                sources.append(
                    str(obj.get(source_column)) if source_column and obj.get(source_column) is not None else None
                )
    return texts, sources


def _load_txt(file_path, **_):
    file_path = Path(file_path).expanduser()
    with open(file_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    if "\n\n" in content:
        texts = [p.strip() for p in content.split("\n\n") if p.strip()]
    else:
        texts = [line.strip() for line in content.splitlines() if line.strip()]

    return texts, [None] * len(texts)


def _load_brat(file_path, **_):
    folder = Path(file_path).expanduser()
    txt_files = sorted(folder.rglob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in: {file_path}")

    print(f"  Found .txt files: {len(txt_files)}")

    texts = []
    for txt_file in txt_files:
        with open(txt_file, encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if content:
            texts.append(content)

    return texts, [None] * len(texts)


def _clean_text(text: str) -> str:
    """Apply minimal whitespace/control-char cleanup while preserving authentic clinical style."""
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_normalized(output_path: str) -> list[dict]:
    """Load normalized seed records for synthetic generation."""
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["texts"]
