#!/usr/bin/env python3
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(ROOT, "output")
CATALOG_PATH = os.path.join(OUTPUT_ROOT, "catalog.json")
VECTOR_LIB_ROOT = os.path.join(OUTPUT_ROOT, "vector_library")


def iter_runs():
    runs = [d for d in os.listdir(OUTPUT_ROOT) if d.startswith("run_")]
    runs.sort()
    for r in runs:
        yield os.path.join(OUTPUT_ROOT, r)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def iter_library_entries(lib_root):
    """
    Scan output/vector_library/{cat}/{concept}/{model}/meta.json (Gd0).
    Npy naming: layer_N.npy (meandiff primary), layer_N_pca.npy (backup).
    """
    for meta_path in sorted(glob.glob(os.path.join(lib_root, "*/*/*/meta.json"))):
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        dir_path = os.path.dirname(meta_path)
        concept    = meta.get("concept", "unknown")
        model_path = meta.get("model_path", "")
        layers     = meta.get("layers", [])

        # Discover primary (meandiff) npy files: match exactly "layer_N.npy"
        npy_files = glob.glob(os.path.join(dir_path, "*.npy"))
        layer_to_path: dict = {}
        for npy_path in npy_files:
            bn = os.path.basename(npy_path)
            m = re.match(r"^layer_(\d+)\.npy$", bn)
            if m:
                layer_idx = int(m.group(1))
                layer_to_path[layer_idx] = npy_path
        vectors = [layer_to_path[l] for l in layers if l in layer_to_path]

        yield {
            "concept":    concept,
            "category":   meta.get("category", ""),
            "run_dir":    dir_path,
            "model_path": model_path,
            "model_name": meta.get("model_name", ""),
            "timestamp":  meta.get("date", ""),
            "layers":     layers,
            "vectors":    vectors,
            "queries":    {},
            "metrics":    {},
            "source":     "vector_library",
        }


def iter_sub_library_entries(lib_root):
    """
    Scan output/vector_library/{cat}/{parent}/sub/{slug}/{model}/meta.json (Gd1+).
    Concept ID = "{parent}/{slug}" to avoid collisions with Gd0 names.
    """
    pattern = os.path.join(lib_root, "*/*/sub/*/*/meta.json")
    for meta_path in sorted(glob.glob(pattern)):
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        dir_path = os.path.dirname(meta_path)
        raw_slug   = meta.get("concept", "unknown")
        # Normalize: old meta.json stored "sub/{slug}", new ones store just "{slug}"
        slug       = raw_slug[4:] if raw_slug.startswith("sub/") else raw_slug
        model_path = meta.get("model_path", "")
        layers     = meta.get("layers", [])

        # Derive parent from path: .../sub/{slug}/{model}/ → parent is two levels up from "sub"
        parts = dir_path.replace("\\", "/").split("/")
        try:
            sub_idx = parts.index("sub")
            parent_concept = parts[sub_idx - 1]
        except (ValueError, IndexError):
            # Fallback: parse from category field ("sensoriale/hot_vs_cold/sub")
            cat_field = meta.get("category", "")
            cat_parts = cat_field.rstrip("/").split("/")
            parent_concept = cat_parts[-2] if len(cat_parts) >= 2 else "unknown"

        concept_id = f"{parent_concept}/{slug}"

        npy_files = glob.glob(os.path.join(dir_path, "*.npy"))
        layer_to_path: dict = {}
        for npy_path in npy_files:
            bn = os.path.basename(npy_path)
            m = re.match(r"^layer_(\d+)\.npy$", bn)
            if m:
                layer_to_path[int(m.group(1))] = npy_path
        vectors = [layer_to_path[l] for l in layers if l in layer_to_path]

        yield {
            "concept":        concept_id,
            "slug":           slug,
            "parent_concept": parent_concept,
            "is_sub_concept": True,
            "gd_level":       1,
            "category":       meta.get("category", "").split("/")[0],
            "run_dir":        dir_path,
            "model_path":     model_path,
            "model_name":     meta.get("model_name", ""),
            "timestamp":      meta.get("date", ""),
            "layers":         layers,
            "vectors":        vectors,
            "queries":        {},
            "metrics":        {},
            "source":         "vector_library_sub",
        }


def main():
    catalog = []
    seen_keys: set = set()

    # --- Gd0: main vector library ---
    if os.path.isdir(VECTOR_LIB_ROOT):
        for entry in iter_library_entries(VECTOR_LIB_ROOT):
            key = (entry["concept"], entry["model_path"])
            seen_keys.add(key)
            catalog.append(entry)

    # --- Gd1+: sub-concept vectors ---
    if os.path.isdir(VECTOR_LIB_ROOT):
        for entry in iter_sub_library_entries(VECTOR_LIB_ROOT):
            key = (entry["concept"], entry["model_path"])
            if key not in seen_keys:
                seen_keys.add(key)
                catalog.append(entry)

    # --- Legacy run_ directories (skip if concept+model already in library) ---
    for run_dir in iter_runs():
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        summary = load_json(summary_path)
        concept = summary.get("summary", {}).get("concept", "unknown")
        model_path_run = summary.get("summary", {}).get("model_path", "")

        # Skip if this (concept, model_path) is already covered by the vector_library
        if (concept, model_path_run) in seen_keys:
            continue

        hot_path = os.path.join(run_dir, "queries_hot.json")
        cold_path = os.path.join(run_dir, "queries_cold.json")
        hot = load_json(hot_path) if os.path.exists(hot_path) else []
        cold = load_json(cold_path) if os.path.exists(cold_path) else []

        deep_layers = summary.get("summary", {}).get("deep_layers", [])

        # Discover npy files by glob and extract layer_idx from actual filenames.
        # Prefer meandiff (no "pca" in name) over PCA variants for same layer.
        npy_files = glob.glob(os.path.join(run_dir, "concept_*.npy"))
        layer_to_path: dict = {}
        for npy_path in npy_files:
            m = re.search(r"_layer_(\d+)\.npy$", npy_path)
            if m:
                layer_idx = int(m.group(1))
                is_pca = "_pca_" in os.path.basename(npy_path)
                existing = layer_to_path.get(layer_idx)
                if existing is None:
                    layer_to_path[layer_idx] = npy_path
                elif is_pca:
                    # Never overwrite a meandiff entry with a PCA one
                    pass
                else:
                    # meandiff always wins
                    layer_to_path[layer_idx] = npy_path
        vectors = [layer_to_path[l] for l in deep_layers if l in layer_to_path]

        entry = {
            "concept": concept,
            "run_dir": run_dir,
            "model_path": model_path_run,
            "timestamp": summary.get("summary", {}).get("timestamp", ""),
            "source": "run",
            "layers": deep_layers,
            "vectors": vectors,
            "queries": {"hot": hot, "cold": cold},
            "metrics": summary.get("results", {}),
        }
        catalog.append(entry)

    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print("Catalog rebuilt")
    print(f"Output: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
