#!/usr/bin/env python3
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")
CATALOG_PATH = os.path.join(OUTPUT_ROOT, "catalog.json")


def load_settings():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def find_latest_run() -> str:
    runs = [d for d in os.listdir(OUTPUT_ROOT) if d.startswith("run_")]
    if not runs:
        raise SystemExit("Nessun run trovato in output/")
    runs.sort()
    return os.path.join(OUTPUT_ROOT, runs[-1])


def load_queries(run_dir: str):
    hot_path = os.path.join(run_dir, "queries_hot.json")
    cold_path = os.path.join(run_dir, "queries_cold.json")
    hot = json.load(open(hot_path)) if os.path.exists(hot_path) else []
    cold = json.load(open(cold_path)) if os.path.exists(cold_path) else []
    return hot, cold


def load_summary(run_dir: str):
    with open(os.path.join(run_dir, "summary.json"), "r") as f:
        return json.load(f)


def main():
    settings = load_settings()
    run_dir = find_latest_run()
    summary = load_summary(run_dir)
    hot, cold = load_queries(run_dir)

    concept_name = summary.get("summary", {}).get("concept", settings.get("concept_name", "hot_vs_cold"))

    entry = {
        "concept": concept_name,
        "run_dir": run_dir,
        "model_path": settings.get("model_path", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layers": summary["summary"]["deep_layers"],
        "vectors": [
            os.path.join(run_dir, f"concept_{concept_name}_layer_{l}.npy")
            for l in summary["summary"]["deep_layers"]
        ],
        "queries": {
            "hot": hot,
            "cold": cold
        },
        "metrics": summary.get("results", {})
    }

    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r") as f:
            catalog = json.load(f)
    else:
        catalog = []

    # Deduplicate: replace existing entry for same (run_dir, concept), else append
    replaced = False
    for i, existing in enumerate(catalog):
        if existing.get("run_dir") == entry["run_dir"] and existing.get("concept") == entry["concept"]:
            catalog[i] = entry
            replaced = True
            break
    if not replaced:
        catalog.append(entry)
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print("Catalog updated")
    print(f"Output: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
