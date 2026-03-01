"""
cosine_matrix.py — Matrice di coseno tra tutti i vettori concept nella vector_library.

Calcola la cosine similarity N×N tra i vettori "migliori" di ogni concept
estratto per un dato modello. Utile per misurare quanto le direzioni siano
ortogonali tra loro — condizione necessaria per un buono steering multiplo
e per confermare che ogni concept cattura qualcosa di distinto.

Selezione del layer migliore per concept:
  - Prima scelta: eval.json["best_layer"]  (held-out SNR, più affidabile)
  - Fallback:     summary.json["results"]  → layer con bootstrap_cos_min massimo

Uso:
  python scripts/cosine_matrix.py --model Gemma2-Uncensored
  python scripts/cosine_matrix.py --model Gemma3-1B-IT --layer-type best_stability
  python scripts/cosine_matrix.py  (tutti e due i modelli, separati)
  python scripts/cosine_matrix.py --model Gemma2-Uncensored --no-plot
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — non serve display
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
VECTOR_LIBRARY = ROOT / "output" / "vector_library"
VECTOR_LIB_ROOT = VECTOR_LIBRARY   # alias per compatibilità con decompose.py
DEFAULT_OUTPUT = ROOT / "output" / "cosine_matrices"


# ── Model slug helpers ─────────────────────────────────────────────────────────

KNOWN_MODELS = {
    "gemma2-uncensored": "Gemma2-Uncensored",
    "gemma3-1b-it":      "Gemma3-1B-IT",
}

def model_name_to_slug(model_name: str) -> str:
    """Converte nome modello in slug directory (lowercase, spazi→trattini)."""
    return model_name.strip().lower().replace(" ", "-")


def slug_to_display(slug: str) -> str:
    """Ritorna il display name dal slug, se noto, altrimenti lo slug stesso."""
    return KNOWN_MODELS.get(slug, slug)


# ── Vector library scanner ────────────────────────────────────────────────────

def scan_concepts(model_slug: str) -> list[dict]:
    """
    Scansiona vector_library e ritorna tutti i concept per cui esiste
    almeno un layer estratto per model_slug.

    Ritorna lista di dict:
      {"concept": str, "category": str, "model_dir": Path}
    """
    found = []
    if not VECTOR_LIBRARY.exists():
        return found
    for cat_dir in sorted(VECTOR_LIBRARY.iterdir()):
        if not cat_dir.is_dir():
            continue
        for concept_dir in sorted(cat_dir.iterdir()):
            if not concept_dir.is_dir():
                continue
            model_dir = concept_dir / model_slug
            if not model_dir.is_dir():
                continue
            # Verifica che ci sia almeno un layer_N.npy (non _pca)
            npy_files = [
                f for f in model_dir.glob("layer_*.npy")
                if "_pca" not in f.name
            ]
            if npy_files:
                found.append({
                    "concept":  concept_dir.name,
                    "category": cat_dir.name,
                    "model_dir": model_dir,
                })
    return found


# ── Best-layer selection ───────────────────────────────────────────────────────

def _best_layer_from_eval(model_dir: Path) -> int | None:
    """Legge eval.json e ritorna best_layer (held-out SNR)."""
    eval_path = model_dir / "eval.json"
    if not eval_path.exists():
        return None
    try:
        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)
        layer = data.get("best_layer")
        return int(layer) if layer is not None else None
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def _best_layer_from_summary(model_dir: Path) -> int | None:
    """
    Fallback: legge summary.json e ritorna il layer con bootstrap_cos_min
    massimo (stabilità convergenza).
    """
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        if not results:
            return None
        best_layer = None
        best_val   = float("-inf")
        for layer_str, layer_data in results.items():
            # boot_min = bootstrap_cos_min nella convergence_pca
            conv = layer_data.get("convergence_pca", {})
            val  = conv.get("bootstrap_cos_min", float("-inf"))
            if val > best_val:
                best_val   = val
                best_layer = int(layer_str)
        return best_layer
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def get_best_layer(model_dir: Path, layer_type: str) -> tuple[int | None, str]:
    """
    Ritorna (layer_idx, source) dove source è "eval" o "summary".
    layer_type: "best_snr" → prova eval prima; "best_stability" → solo summary.
    """
    if layer_type == "best_stability":
        layer = _best_layer_from_summary(model_dir)
        if layer is not None:
            return layer, "summary"
        layer = _best_layer_from_eval(model_dir)
        if layer is not None:
            return layer, "eval(fallback)"
        return None, "none"
    else:  # best_snr (default)
        layer = _best_layer_from_eval(model_dir)
        if layer is not None:
            return layer, "eval"
        layer = _best_layer_from_summary(model_dir)
        if layer is not None:
            return layer, "summary(fallback)"
        # Ultimo fallback: prendi il layer numerico più alto disponibile
        npy_files = sorted(
            [f for f in model_dir.glob("layer_*.npy") if "_pca" not in f.name],
            key=lambda p: int(p.stem.replace("layer_", "")),
        )
        if npy_files:
            layer = int(npy_files[-1].stem.replace("layer_", ""))
            return layer, "last_available"
        return None, "none"


# ── Vector loading ─────────────────────────────────────────────────────────────

def load_vector(model_dir: Path, layer_idx: int) -> np.ndarray | None:
    """Carica layer_N.npy (non _pca). Ritorna None se non trovato."""
    npy_path = model_dir / f"layer_{layer_idx}.npy"
    if not npy_path.exists():
        print(f"  [WARN] Vettore non trovato: {npy_path}")
        return None
    try:
        vec = np.load(npy_path)
        vec = vec.flatten().astype(np.float64)
        return vec
    except Exception as e:
        print(f"  [WARN] Errore caricando {npy_path}: {e}")
        return None


# ── Cosine similarity ──────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity tra due vettori 1D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Costruisce matrice N×N di cosine similarity."""
    n = len(vectors)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_similarity(vectors[i], vectors[j])
    return mat


# ── ASCII table ────────────────────────────────────────────────────────────────

def _short_label(concept: str, max_len: int = 6) -> str:
    """Abbreviazione di un concept per la tabella ASCII."""
    # Prende la prima parola prima di "_vs_"
    parts = concept.split("_vs_")
    label = parts[0].replace("_", "")
    return label[:max_len]


def print_matrix_table(concepts: list[str], matrix: np.ndarray, layers_used: dict):
    """Stampa tabella ASCII della matrice."""
    n = len(concepts)
    labels = [_short_label(c) for c in concepts]
    col_w  = 7   # larghezza colonna

    # Header colonne
    row_label_w = 26
    header = " " * row_label_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(" " * row_label_w + "-" * (col_w * n))

    for i, concept in enumerate(concepts):
        row_name = f"{concept:<{row_label_w - 2}}"
        row_vals = "".join(f"{matrix[i, j]:>{col_w}.3f}" for j in range(n))
        print(f"{row_name}  {row_vals}")

    # Off-diagonal stats
    off_diag = []
    off_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag.append(matrix[i, j])
                if i < j:
                    off_pairs.append((matrix[i, j], concepts[i], concepts[j]))

    if off_diag:
        print()
        min_val, min_a, min_b = min(off_pairs, key=lambda x: x[0])
        max_val, max_a, max_b = max(off_pairs, key=lambda x: x[0])
        mean_val = float(np.mean(off_diag))
        print(f"Min off-diagonal: {min_val:.3f}  ({min_a} <-> {min_b})")
        print(f"Max off-diagonal: {max_val:.3f}  ({max_a} <-> {max_b})")
        print(f"Mean off-diagonal: {mean_val:.3f}")


# ── Heatmap plot ───────────────────────────────────────────────────────────────

def save_heatmap(
    concepts: list[str],
    matrix: np.ndarray,
    model_display: str,
    layer_type: str,
    output_path: Path,
):
    """Salva heatmap matplotlib con annotazioni numeriche."""
    n = len(concepts)
    fig_size = max(7, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")

    # Annotazioni numeriche
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    # Labels assi
    short_labels = [c.replace("_vs_", "\nvs\n") for c in concepts]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    ax.set_title(
        f"Cosine Matrix — {model_display}\n"
        f"Layer selection: {layer_type}  |  {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=11,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap salvata: {output_path}")


# ── JSON output ────────────────────────────────────────────────────────────────

def save_json(
    concepts: list[str],
    matrix: np.ndarray,
    model_display: str,
    layer_type: str,
    layers_used: dict,
    layer_sources: dict,
    output_path: Path,
):
    data = {
        "model":        model_display,
        "layer_type":   layer_type,
        "concepts":     concepts,
        "layers_used":  layers_used,
        "layer_sources": layer_sources,
        "matrix":       matrix.tolist(),
        "timestamp":    datetime.now().isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON salvato:    {output_path}")


# ── Core logic ─────────────────────────────────────────────────────────────────

def run_for_model(
    model_slug:  str,
    layer_type:  str,
    output_dir:  Path,
    no_plot:     bool,
):
    model_display = slug_to_display(model_slug)
    print(f"\n{'='*62}")
    print(f"  Cosine Matrix — {model_display}")
    print(f"{'='*62}")

    # Scansione concept
    entries = scan_concepts(model_slug)
    if not entries:
        print(f"[WARN] Nessun concept trovato per '{model_slug}' in {VECTOR_LIBRARY}")
        return

    print(f"Concept trovati: {len(entries)}")

    # Carica vettori
    concepts     = []
    vectors      = []
    layers_used  = {}
    layer_sources = {}

    for entry in entries:
        concept   = entry["concept"]
        model_dir = entry["model_dir"]

        layer_idx, source = get_best_layer(model_dir, layer_type)
        if layer_idx is None:
            print(f"  [SKIP] {concept}: nessun layer selezionabile — skippato")
            continue

        vec = load_vector(model_dir, layer_idx)
        if vec is None:
            print(f"  [SKIP] {concept}: vettore layer_{layer_idx}.npy non caricabile")
            continue

        concepts.append(concept)
        vectors.append(vec)
        layers_used[concept]  = layer_idx
        layer_sources[concept] = source
        print(f"  + {concept:<30} L{layer_idx}  ({source})")

    if len(concepts) < 2:
        print(f"[ERROR] Serve almeno 2 concept per la matrice. Trovati: {len(concepts)}")
        return

    # Verifica dimensioni coerenti (warning se diverse, non blocca)
    dims = [v.shape[0] for v in vectors]
    if len(set(dims)) > 1:
        print(f"  [WARN] Dimensioni vettori disomogenee: {dict(zip(concepts, dims))}")

    # Matrice
    n   = len(concepts)
    mat = build_matrix(vectors)

    # Layer summary per stampa
    layer_str = ", ".join(f"{c}->L{layers_used[c]}" for c in concepts)
    print(f"\nConcepts ({n}): {', '.join(concepts)}")
    print(f"Layer per concept: {layer_str}")
    print()

    # Stampa tabella ASCII
    print_matrix_table(concepts, mat, layers_used)
    print()

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name   = f"{model_slug}_matrix"
    json_path   = output_dir / f"{base_name}.json"
    png_path    = output_dir / f"{base_name}.png"

    save_json(
        concepts, mat, model_display, layer_type,
        {c: layers_used[c] for c in concepts},
        {c: layer_sources[c] for c in concepts},
        json_path,
    )

    if not no_plot:
        if HAS_MATPLOTLIB:
            save_heatmap(concepts, mat, model_display, layer_type, png_path)
        else:
            print("  [INFO] matplotlib non disponibile — skip heatmap (installa con: pip install matplotlib)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calcola matrice coseno tra concept nella vector_library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help=(
            "Nome modello (es. 'Gemma2-Uncensored' o 'Gemma3-1B-IT'). "
            "Se omesso, elabora tutti i modelli trovati nella vector_library."
        ),
    )
    parser.add_argument(
        "--layer-type",
        choices=["best_snr", "best_stability"],
        default="best_snr",
        help=(
            "Criterio selezione layer: "
            "'best_snr' usa eval.json (held-out SNR, preferito); "
            "'best_stability' usa summary.json (bootstrap_cos_min massimo)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory output per JSON e PNG.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Salta il salvataggio dell'heatmap matplotlib.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.model:
        # Singolo modello
        slug = model_name_to_slug(args.model)
        run_for_model(slug, args.layer_type, output_dir, args.no_plot)
    else:
        # Tutti i modelli trovati nella vector_library
        slugs_found: set[str] = set()
        if VECTOR_LIBRARY.exists():
            for cat_dir in VECTOR_LIBRARY.iterdir():
                if not cat_dir.is_dir():
                    continue
                for concept_dir in cat_dir.iterdir():
                    if not concept_dir.is_dir():
                        continue
                    for model_dir in concept_dir.iterdir():
                        if model_dir.is_dir():
                            npy = [
                                f for f in model_dir.glob("layer_*.npy")
                                if "_pca" not in f.name
                            ]
                            if npy:
                                slugs_found.add(model_dir.name)

        if not slugs_found:
            print(f"[ERROR] Nessun modello trovato in {VECTOR_LIBRARY}")
            sys.exit(1)

        print(f"Modelli trovati: {sorted(slugs_found)}")
        for slug in sorted(slugs_found):
            run_for_model(slug, args.layer_type, output_dir, args.no_plot)

    print("\nDone.")


if __name__ == "__main__":
    main()
