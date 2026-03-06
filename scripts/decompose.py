"""
decompose.py — Orchestratore del loop di auto-analisi ricorsivo.

Fase 2 — Sub-Concept Decomposition (Project Jedi)
Vedi: experiments/03_sub_concept_decomposition.md

Loop chiuso:
  1. M40 propone sub-concetti (concept_expander step1)
  2. M40 genera dataset chirurgici (concept_expander step2)
  3. MI50 estrae vettori (probe_concept.py via subprocess)
  4. Analisi geometrica separabilità (cosine_matrix)
  5. M40 giudica separabilità reale da steering (sub_concept_eval)
  → Se non validati: feedback a M40, nuova iterazione (max 3)
  → Se validati: archivia e opzionalmente ricorre sui sub-vettori

Uso:
  python scripts/decompose.py --concept hot_vs_cold
  python scripts/decompose.py --concept hot_vs_cold --model Gemma2-Uncensored --depth 2
  python scripts/decompose.py --concept hot_vs_cold --dry-run
  python scripts/decompose.py --concept hot_vs_cold --start-from-version 2  # riprende dal ciclo 2
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
SCRIPTS_DIR      = ROOT / "scripts"
PYTHON           = ROOT / ".venv" / "bin" / "python"
SUB_CONCEPTS_DIR = ROOT / "config" / "sub_concepts"
VECTOR_LIB_ROOT  = ROOT / "output" / "vector_library"
DECOMPOSE_LOG    = ROOT / "output" / "decompose_runs"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL        = "Gemma2-Uncensored"
DEFAULT_STEERING_URL = "http://localhost:8010"
DEFAULT_M40_URL      = "http://localhost:11435"
DEFAULT_MAX_ITER     = 3       # max iterazioni per livello prima di dichiarare "limite semantico"
DEFAULT_MAX_DEPTH    = 2       # profondità massima ricorsione (0=broad, 1=sub, 2=sub-sub)
DEFAULT_GAIN         = 200
DEFAULT_ALPHA        = 1.0
DEFAULT_N_PROMPTS    = 3


# ── Import moduli del progetto ─────────────────────────────────────────────────
sys.path.insert(0, str(SCRIPTS_DIR))

from concept_expander import step1_analyze, step2_generate, _load_meta
from sub_concept_eval  import run_eval as run_separation_eval
from gpu_utils import gpu_prepare_for_probe, gpu_restore_after_probe, check_m40_on_gpu
from cosine_matrix     import (
    scan_concepts, get_best_layer, load_vector,
    build_matrix, print_matrix_table,
    save_heatmap, save_json as save_matrix_json,
    model_name_to_slug, VECTOR_LIB_ROOT as CM_VECTOR_LIB,
)


# ── Logging strutturato ────────────────────────────────────────────────────────

class DecomposeLogger:
    """Log strutturato di ogni run di decompose. Salvato come JSONL."""

    def __init__(self, concept: str, model: str, depth: int):
        DECOMPOSE_LOG.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = model_name_to_slug(model)
        self.path = DECOMPOSE_LOG / f"{ts}_{concept}_{slug}_d{depth}.jsonl"
        self.concept = concept
        self.model   = model

    def log(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def info(self, msg: str, **kwargs):
        print(f"  {msg}")
        return self.log({"type": "info", "msg": msg, **kwargs})

    def step(self, n: int, name: str):
        hdr = f"\n{'━'*60}\n  STEP {n} — {name}\n{'━'*60}"
        print(hdr)
        return self.log({"type": "step", "step": n, "name": name})

    def result(self, **kwargs):
        return self.log({"type": "result", **kwargs})


# ── Step 3: probe via subprocess ───────────────────────────────────────────────

def run_probe_for_sub(
    sub_slug: str,
    parent_concept: str,
    category: str,
    model: str,
    logger: DecomposeLogger,
    dry_run: bool = False,
) -> bool:
    """
    Estrae il vettore per un sub-concetto chiamando probe_concept.py come subprocess.
    Il sub-concept JSON deve essere già in config/sub_concepts/{parent}/{sub_slug}.json.
    Output in: output/vector_library/{category}/{parent}/{model_slug}/sub/{sub_slug}/
    """
    json_path = SUB_CONCEPTS_DIR / parent_concept / f"{sub_slug}.json"
    if not json_path.exists():
        logger.info(f"  SKIP probe {sub_slug} — JSON non trovato: {json_path}")
        return False

    # Patch temporanea: probe_concept.py salva in vector_library/{category}/{concept}/{model}/
    # Per i sub-concetti vogliamo un path diverso. Usiamo la categoria del parent
    # e creiamo un concept "virtuale" nel formato {parent}/{model_slug}/sub/{sub_slug}
    # attraverso un campo extra nel JSON
    with open(json_path) as f:
        data = json.load(f)

    # probe_concept.py usa normalize_name_for_path() che rimuove "/".
    # Quindi concept deve essere solo lo slug (senza prefix "sub/"),
    # e il path sub/ si ottiene mettendolo nella category.
    data["category"] = f"{category}/{parent_concept}/sub"
    data["concept"]  = sub_slug

    # Salva JSON patchato temporaneamente
    patched_path = json_path.parent / f"_tmp_{sub_slug}.json"
    with open(patched_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    cmd = [str(PYTHON), str(SCRIPTS_DIR / "probe_concept.py"),
           "--concept", str(patched_path),
           "--model", model,
           "--eval"]

    logger.info(f"  Probe: {sub_slug} / {model}")
    if dry_run:
        logger.info(f"  [dry-run] cmd: {' '.join(cmd)}")
        patched_path.unlink(missing_ok=True)
        return True

    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
        patched_path.unlink(missing_ok=True)
        if result.returncode != 0:
            logger.info(f"  ERRORE probe {sub_slug}: {result.stderr[-200:]}")
            return False
        logger.info(f"  Probe OK: {sub_slug}")
        return True
    except subprocess.TimeoutExpired:
        logger.info(f"  TIMEOUT probe {sub_slug}")
        patched_path.unlink(missing_ok=True)
        return False


# ── Step 4: cosine matrix solo per i sub di questo parent ─────────────────────

def run_cosine_step(
    parent_concept: str,
    sub_slugs: list[str],
    model: str,
    category: str,
    logger: DecomposeLogger,
) -> dict:
    """
    Calcola matrice coseno solo tra i sub-vettori del parent (+ il broad stesso).
    Ritorna dict con matrix e interpretazione.
    """
    model_slug = model_name_to_slug(model)

    vectors = {}
    layers_used = {}

    # Vettore broad
    broad_dirs = list(VECTOR_LIB_ROOT.glob(f"*/{parent_concept}/{model_slug}"))
    if broad_dirs:
        broad_dir = broad_dirs[0]
        layer, source = get_best_layer(broad_dir, "best_snr")
        if layer is not None:
            vec = load_vector(broad_dir, layer)
            if vec is not None:
                vectors[parent_concept] = vec
                layers_used[parent_concept] = {"layer": layer, "source": source}

    # Sub-vettori (path: {category}/{parent}/sub/{slug}/{model_slug}/)
    for slug in sub_slugs:
        sub_dirs = list(VECTOR_LIB_ROOT.glob(
            f"{category}/{parent_concept}/sub/{slug}/{model_slug}"
        ))
        if not sub_dirs:
            logger.info(f"  Sub-vettore non trovato: {slug}")
            continue
        sub_dir = sub_dirs[0]
        layer, source = get_best_layer(sub_dir, "best_snr")
        if layer is not None:
            vec = load_vector(sub_dir, layer)
            if vec is not None:
                vectors[slug] = vec
                layers_used[slug] = {"layer": layer, "source": source}

    if len(vectors) < 2:
        logger.info("  Cosine matrix: troppo pochi vettori disponibili")
        return {}

    names = list(vectors.keys())
    vecs  = [vectors[n] for n in names]
    matrix = build_matrix(vecs)

    print_matrix_table(names, matrix, layers_used)

    # Statistiche
    n = len(names)
    off_diag = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
    stats = {
        "min": round(float(min(off_diag)), 4),
        "max": round(float(max(off_diag)), 4),
        "mean": round(float(sum(off_diag) / len(off_diag)), 4),
    }
    logger.result(step=4, concepts=names, matrix=matrix.tolist(),
                  layers_used=layers_used, stats=stats)

    # Salva heatmap e JSON
    out_dir = DECOMPOSE_LOG
    layers_by_concept  = {k: v["layer"]  for k, v in layers_used.items()}
    sources_by_concept = {k: v["source"] for k, v in layers_used.items()}
    save_matrix_json(
        concepts=names,
        matrix=matrix,
        model_display=model,
        layer_type="best_snr",
        layers_used=layers_by_concept,
        layer_sources=sources_by_concept,
        output_path=out_dir / f"cosine_{parent_concept}_{model_slug}.json",
    )
    save_heatmap(
        concepts=names,
        matrix=matrix,
        model_display=model,
        layer_type="best_snr",
        output_path=out_dir / f"cosine_{parent_concept}_{model_slug}.png",
    )

    return {"concepts": names, "matrix": matrix.tolist(),
            "layers_used": layers_used, "stats": stats}


# ── Loop principale ────────────────────────────────────────────────────────────

def decompose_concept(
    concept: str,
    model: str,
    category: str,
    steering_url: str,
    m40_url: str,
    depth: int,
    max_depth: int,
    max_iter: int,
    gain: int,
    alpha: float,
    n_prompts: int,
    dry_run: bool,
    start_version: int,
    logger: DecomposeLogger,
) -> dict:
    """
    Loop ricorsivo di decomposizione per un singolo concetto.
    Ritorna dict con risultato: validated_subs, semantic_limit, needs_refinement.
    """
    indent = "  " * depth
    print(f"\n{indent}{'='*58}")
    print(f"{indent}  DECOMPOSE: {concept}  (depth={depth})")
    print(f"{indent}{'='*58}")

    logger.log({"type": "decompose_start", "concept": concept, "depth": depth})

    validated_subs = []
    semantic_limit = False

    for iteration in range(start_version, start_version + max_iter):
        version = iteration
        print(f"\n{indent}[Iterazione {iteration - start_version + 1}/{max_iter} — v{version}]")

        # ── Step 1: M40 propone sub-concetti ──────────────────────────────────
        logger.step(1, f"Analisi M40 → sub-concetti (v{version})")

        # Carica feedback precedente se esiste
        feedback = None
        if version > start_version:
            prev_eval = (ROOT / "output" / "sub_concept_evals" /
                         concept / model_name_to_slug(model) / f"eval_v{version-1}.json")
            if prev_eval.exists():
                with open(prev_eval) as f:
                    prev = json.load(f)
                verdict  = prev.get("verdict", {})
                feedback = {
                    "needs_refinement": verdict.get("needs_refinement", []),
                    "refinement_suggestions": verdict.get("refinement_suggestions", {}),
                    "overall_assessment": verdict.get("overall_assessment", ""),
                }
                logger.info(f"  Feedback v{version-1}: {feedback['needs_refinement']}")

        meta = step1_analyze(
            concept=concept,
            model=model,
            m40_url=m40_url,
            version=version,
            dry_run=dry_run,
            feedback=feedback,
        )

        sub_concepts = meta.get("sub_concepts", [])
        sub_slugs    = [s["slug"] for s in sub_concepts]
        logger.info(f"  Proposti: {sub_slugs}")

        # ── Step 2: M40 genera dataset chirurgici ──────────────────────────────
        logger.step(2, f"Generazione dataset chirurgici (v{version})")
        step2_generate(
            concept=concept,
            model=model,
            m40_url=m40_url,
            version=version,
            dry_run=dry_run,
            meta=meta if meta else None,
        )

        # ── Step 3: MI50 estrae vettori ────────────────────────────────────────
        logger.step(3, f"Estrazione vettori — MI50")
        probe_ok = []
        if not dry_run:
            # Controlla GPU, aspetta idle, scarica modello, verifica VRAM libera
            gpu_prepare_for_probe(steering_url, log=logger.info)

        for slug in sub_slugs:
            ok = run_probe_for_sub(slug, concept, category, model, logger, dry_run)
            if ok:
                probe_ok.append(slug)

        if not dry_run:
            # Ricarica il modello (con verifica stato GPU prima del load)
            loaded = gpu_restore_after_probe(steering_url, model, log=logger.info)
            if not loaded:
                logger.info(f"  [warn] Reload {model} non confermato — continuo comunque")

        if not probe_ok:
            logger.info("  Nessun vettore estratto. Salto i prossimi step.")
            continue

        # ── Step 4: Cosine matrix ──────────────────────────────────────────────
        logger.step(4, "Analisi geometrica separabilità")
        cosine_result = run_cosine_step(concept, probe_ok, model, category, logger)

        # ── Step 5: M40 giudica separabilità reale ────────────────────────────
        logger.step(5, "Steering test + giudizio M40")

        if not dry_run:
            eval_result = run_separation_eval(
                parent_concept=concept,
                version=version,
                model_name=model,
                steering_url=steering_url,
                m40_url=m40_url,
                gain=gain,
                alpha=alpha,
                max_tokens=120,
                n_prompts=n_prompts,
            )
            verdict  = eval_result.get("verdict", {})
            all_ok   = verdict.get("all_validated", False)
            validated_this = verdict.get("validated_concepts", [])
            needs_ref      = verdict.get("needs_refinement", [])
        else:
            logger.info("  [dry-run] skip steering eval")
            all_ok = False
            validated_this = []
            needs_ref = sub_slugs

        logger.result(
            step=5, version=version, all_validated=all_ok,
            validated=validated_this, needs_refinement=needs_ref,
        )

        if all_ok:
            validated_subs = probe_ok
            print(f"\n{indent}  ✓ VALIDATI dopo {iteration - start_version + 1} iterazione/i")
            break
        else:
            remaining = [s for s in needs_ref if s in probe_ok]
            print(f"\n{indent}  ✗ Non validati: {needs_ref}")
            if iteration < start_version + max_iter - 1:
                print(f"{indent}  → Prossima iterazione con feedback M40")
            else:
                semantic_limit = True
                print(f"\n{indent}  ⚠ Max iterazioni raggiunto — limite semantico del modello")
                validated_subs = validated_this  # salva quelli parzialmente buoni

    # ── Ricorsione sui sub validati ────────────────────────────────────────────
    if validated_subs and depth < max_depth and not dry_run:
        print(f"\n{indent}  → Ricorsione su {len(validated_subs)} sub-vettori validati (depth {depth+1})")
        sub_results = {}
        for sub_slug in validated_subs:
            sub_results[sub_slug] = decompose_concept(
                concept     = sub_slug,
                model       = model,
                category    = f"{category}/{concept}",
                steering_url = steering_url,
                m40_url     = m40_url,
                depth       = depth + 1,
                max_depth   = max_depth,
                max_iter    = max_iter,
                gain        = gain,
                alpha       = alpha,
                n_prompts   = n_prompts,
                dry_run     = dry_run,
                start_version = 1,
                logger      = logger,
            )
    else:
        sub_results = {}

    result = {
        "concept": concept,
        "depth": depth,
        "validated_subs": validated_subs,
        "semantic_limit": semantic_limit,
        "sub_results": sub_results,
    }
    logger.log({"type": "decompose_end", **result})
    return result


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Loop di auto-analisi ricorsivo per sub-concept decomposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--concept",     required=True,
                        help="Concept da decomporre (es. hot_vs_cold)")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--category",    default="sensoriale",
                        help="Categoria nella vector_library (es. sensoriale, termico)")
    parser.add_argument("--depth",       type=int, default=0,
                        help="Profondità di partenza (0=broad)")
    parser.add_argument("--max-depth",   type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-iter",    type=int, default=DEFAULT_MAX_ITER,
                        help="Max iterazioni per livello prima di dichiarare limite semantico")
    parser.add_argument("--gain",        type=int, default=DEFAULT_GAIN)
    parser.add_argument("--alpha",       type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n-prompts",   type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument("--steering-url", default=DEFAULT_STEERING_URL)
    parser.add_argument("--m40-url",     default=DEFAULT_M40_URL)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Mostra prompt e struttura senza chiamare M40 né fare probe")
    parser.add_argument("--start-from-version", type=int, default=1,
                        help="Riprendi il loop dalla versione N (utile se step1/2 già fatti)")

    args = parser.parse_args()

    logger = DecomposeLogger(args.concept, args.model, args.depth)
    logger.log({
        "type": "run_start",
        "concept": args.concept, "model": args.model,
        "max_depth": args.max_depth, "max_iter": args.max_iter,
        "dry_run": args.dry_run,
    })

    print(f"\n{'='*60}")
    print(f"  DECOMPOSE — {args.concept}")
    print(f"  Modello  : {args.model}")
    print(f"  Max depth: {args.max_depth}  Max iter/livello: {args.max_iter}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Log      : {logger.path.name}")
    print(f"{'='*60}")

    # Verifica GPU M40 prima di partire (blocca se su CPU)
    if not args.dry_run:
        print("\n[pre-flight] Verifica M40 GPU...")
        try:
            check_m40_on_gpu(m40_url=args.m40_url, log=print)
        except RuntimeError as e:
            print(f"\nBLOCCATO: {e}")
            sys.exit(1)
        print()

    t0 = time.time()
    result = decompose_concept(
        concept      = args.concept,
        model        = args.model,
        category     = args.category,
        steering_url = args.steering_url,
        m40_url      = args.m40_url,
        depth        = args.depth,
        max_depth    = args.max_depth,
        max_iter     = args.max_iter,
        gain         = args.gain,
        alpha        = args.alpha,
        n_prompts    = args.n_prompts,
        dry_run      = args.dry_run,
        start_version = args.start_from_version,
        logger       = logger,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETATO in {elapsed/60:.1f} min")
    print(f"  Validati   : {result['validated_subs']}")
    print(f"  Limite sem.: {result['semantic_limit']}")
    print(f"  Log        : {logger.path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
