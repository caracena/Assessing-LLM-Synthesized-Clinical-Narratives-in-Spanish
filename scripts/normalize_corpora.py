"""Normalize all clinical seed corpora defined in corpus_config_v2.yaml.

Uses deterministic config hashing via corpus_normalizer.is_normalized_valid so
up-to-date outputs are skipped cleanly and restricted local corpora absent on an
external machine are skipped with a clear notice.

Usage:
    python scripts/normalize_corpora.py
    python scripts/normalize_corpora.py --config corpus_config_v2.yaml
    python scripts/normalize_corpora.py --only atadolcc cantemist
    python scripts/normalize_corpora.py --check
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from corpus_normalizer import compute_config_hash, is_normalized_valid, normalize_corpus

BASE_DIR = Path(__file__).parent.parent
POR_SOURCE_DIR = BASE_DIR / "corpus" / "por_source"
CONFIG_DEFAULT = BASE_DIR / "corpus_config_v2.yaml"


def _expand_corpora(corpora: list) -> list[dict]:
    """Expand YAML entries (monolithic or grouped splits) into flat normalization jobs."""
    jobs = []
    for entry in corpora:
        base = {
            "loader": entry.get("loader") or entry.get("source"),
            "repo": entry.get("repo"),
            "file_path": entry.get("file_path"),
            "text_column": entry.get("text_column", "text"),
            "source_column": entry.get("source_column"),
            "hf_split": entry.get("hf_split"),
            "hf_subset": entry.get("hf_subset"),
            "min_text_length": entry.get("min_text_length", 10),
            "max_text_length": entry.get("max_text_length", 50000),
        }

        if "splits" in entry:
            group_name = entry.get("repo_group", entry.get("repo", "group"))
            for split in entry["splits"]:
                job = {
                    **base,
                    "name": split["name"],
                    "include_sources": split.get("include_sources"),
                    "_group": group_name,
                }
                job.update({k: v for k, v in split.items() if k not in ("name", "include_sources")})
                jobs.append(job)
        else:
            jobs.append({
                **base,
                "name": entry["name"],
                "include_sources": entry.get("include_sources"),
                "_group": None,
            })

    return jobs


def check_corpora(config_path: str = None) -> dict:
    """Inspect YAML corpus entries and report whether outputs are pending, current, or stale."""
    config_path = Path(config_path or CONFIG_DEFAULT)
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}")
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    jobs = _expand_corpora(config.get("corpora", []))

    print(f"\n{'='*70}")
    print(f"CORPUS CHECK -- {len(jobs)} seed corpora defined in YAML")
    print(f"{'='*70}")
    print(f"{'Name':<28} {'Status':<16} {'Detail'}")
    print(f"{'-'*70}")

    results = {}
    for job in jobs:
        name = job["name"]
        output_path = str(POR_SOURCE_DIR / f"{name}_normalizado.json")
        expected_hash = compute_config_hash(job)
        p = Path(output_path)

        if not p.exists():
            if job.get("loader") == "brat" and job.get("file_path") and not Path(job["file_path"]).expanduser().exists():
                status, detail = "SKIP (LOCAL)", f"restricted corpus missing at {job['file_path']}"
            else:
                status, detail = "PENDING", "not normalized yet"
        elif is_normalized_valid(output_path, expected_hash):
            with open(p, encoding="utf-8") as f:
                n = json.load(f)["metadata"]["n_normalized"]
            status, detail = "UP TO DATE", f"{n} texts"
        else:
            status, detail = "STALE", "config changed -- will regenerate"

        print(f"{name:<28} {status:<16} {detail}")
        results[name] = {"status": status, "detail": detail, "output_path": output_path}

    print(f"{'='*70}\n")
    return results


def normalize_all(config_path: str = None, force_names: set = None, only_names: set = None):
    """Normalize pending or stale corpora defined in corpus_config_v2.yaml."""
    config_path = Path(config_path or CONFIG_DEFAULT)
    force_names = force_names or set()

    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}")
        return []

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    jobs = _expand_corpora(config.get("corpora", []))
    if only_names:
        jobs = [j for j in jobs if j["name"] in only_names]

    print(f"\n{'='*60}")
    print("NORMALIZE CORPORA")
    print(f"Seed corpora to process: {len(jobs)}")
    print(f"{'='*60}\n")

    POR_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    skipped_missing = 0
    for job in jobs:
        name = job["name"]
        group = job.pop("_group", None)
        output_path = str(POR_SOURCE_DIR / f"{name}_normalizado.json")

        # Skip restricted local corpora gracefully if absent on external machines
        if job.get("loader") == "brat":
            fp = job.get("file_path")
            if fp and not Path(fp).expanduser().exists():
                print(f"[SKIP] Restricted local seed corpus '{name}' not found at {fp}; skipping.")
                skipped_missing += 1
                continue

        if group:
            print(f"[{group}] {name}")
        else:
            print(f"{name}")

        result = normalize_corpus(
            output_path=output_path,
            force=name in force_names,
            **job,
        )
        results.append(result)

    n_generated = sum(1 for r in results if not r["skipped"])
    n_cached = sum(1 for r in results if r["skipped"])

    print(f"\n{'='*60}")
    print("[OK] Normalization complete")
    print(f"  Generated/updated:       {n_generated}")
    print(f"  Already up to date:      {n_cached}")
    if skipped_missing:
        print(f"  Skipped (missing local): {skipped_missing}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize clinical seed corpora from YAML configuration")
    parser.add_argument("--config", default=None, help="Path to corpus_config_v2.yaml")
    parser.add_argument("--check", action="store_true", help="Inspect normalization status without writing files")
    parser.add_argument("--force", nargs="*", default=None, help="Corpus names to force-regenerate")
    parser.add_argument("--only", nargs="+", default=None, help="Normalize only the specified corpus names")
    args = parser.parse_args()

    if args.check:
        check_corpora(args.config)
    else:
        normalize_all(
            args.config,
            force_names=set(args.force) if args.force else None,
            only_names=set(args.only) if args.only else None,
        )
