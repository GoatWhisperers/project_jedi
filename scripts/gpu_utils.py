"""
gpu_utils.py — Gestione sicura della GPU per Project Jedi.

Sequenza corretta per caricare un modello sulla GPU:
  1. Controlla cosa c'è sulla GPU e se sta ancora lavorando
  2. Aspetta che finisca (se gpu_use_pct > soglia)
  3. Unload pulito via steering server
  4. Verifica che la VRAM si sia svuotata
  5. Carica il nuovo modello
  6. Verifica che sia pronto

Uso:
    from gpu_utils import gpu_prepare, gpu_reload

    # Prima del probe (libera VRAM):
    gpu_prepare(steering_url="http://localhost:8010")

    # Dopo il probe (ricarica modello):
    gpu_reload(steering_url="http://localhost:8010", model_name="Gemma2-Uncensored")
"""

import json
import time
import urllib.error
import urllib.request
from typing import Optional

# ── Soglie ─────────────────────────────────────────────────────────────────────
GPU_BUSY_THRESHOLD_PCT  = 15    # % GPU use oltre il quale aspettiamo
VRAM_FREE_THRESHOLD_GB  = 3.0   # VRAM libera minima dopo unload (MI50 ha 32GB)
WAIT_BUSY_TIMEOUT_S     = 120   # max attesa GPU idle
WAIT_VRAM_TIMEOUT_S     = 60    # max attesa VRAM libera dopo unload
WAIT_LOAD_TIMEOUT_S     = 180   # max attesa caricamento modello


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _post(url: str, body: dict, timeout: int = 30) -> Optional[dict]:
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ── Funzioni pubbliche ─────────────────────────────────────────────────────────

def get_gpu_status(steering_url: str) -> dict:
    """Ritorna lo stato attuale della GPU: vram, gpu_use, modello attivo."""
    gpu  = _get(f"{steering_url}/api/gpu")   or {}
    mods = _get(f"{steering_url}/api/models") or {}
    return {
        "active_model":  mods.get("active", ""),
        "gpu_use_pct":   gpu.get("gpu_use_pct", 0),
        "vram_used_gb":  gpu.get("vram_used_gb", 0),
        "vram_total_gb": gpu.get("vram_total_gb", 0),
    }


def wait_gpu_idle(steering_url: str, log=print) -> bool:
    """
    Aspetta che la GPU non stia più elaborando (gpu_use_pct < soglia).
    Ritorna True se idle entro il timeout, False altrimenti.
    """
    deadline = time.time() + WAIT_BUSY_TIMEOUT_S
    while time.time() < deadline:
        st = get_gpu_status(steering_url)
        use = st.get("gpu_use_pct", 0)
        if use < GPU_BUSY_THRESHOLD_PCT:
            return True
        log(f"  [gpu] GPU occupata al {use:.0f}% — attendo...")
        time.sleep(5)
    log(f"  [gpu] TIMEOUT attesa GPU idle dopo {WAIT_BUSY_TIMEOUT_S}s")
    return False


def gpu_unload(steering_url: str, log=print) -> bool:
    """
    Sequenza completa di unload:
      1. Legge stato attuale
      2. Aspetta che la GPU sia idle
      3. Chiama /api/unload_model
      4. Verifica che la VRAM si sia liberata
    Ritorna True se OK.
    """
    st = get_gpu_status(steering_url)
    active = st.get("active_model", "")
    vram   = st.get("vram_used_gb", 0)

    if not active:
        log(f"  [gpu] Nessun modello caricato — VRAM usata: {vram:.1f} GB")
        return True

    log(f"  [gpu] Modello attivo: {active} — VRAM: {vram:.1f} GB")

    # 1. Aspetta idle
    log("  [gpu] Attendo GPU idle...")
    wait_gpu_idle(steering_url, log=log)

    # 2. Unload (endpoint GET nel steering server)
    log("  [gpu] Unload modello...")
    result = _get(f"{steering_url}/api/unload_model")
    if not result or not result.get("ok"):
        log("  [gpu] WARN: unload_model non ha risposto OK")

    # 3. Aspetta VRAM libera
    deadline = time.time() + WAIT_VRAM_TIMEOUT_S
    while time.time() < deadline:
        st = get_gpu_status(steering_url)
        vram_now = st.get("vram_used_gb", 0)
        total    = st.get("vram_total_gb", 1)
        vram_free = total - vram_now
        if vram_free >= VRAM_FREE_THRESHOLD_GB or not st.get("active_model"):
            log(f"  [gpu] VRAM liberata — usata: {vram_now:.1f} GB / {total:.1f} GB")
            return True
        log(f"  [gpu] Attendo VRAM libera... ({vram_now:.1f}/{total:.1f} GB)")
        time.sleep(3)

    log(f"  [gpu] WARN: VRAM non scesa sotto soglia entro {WAIT_VRAM_TIMEOUT_S}s — continuo comunque")
    return False


def gpu_load(steering_url: str, model_name: str, log=print) -> bool:
    """
    Carica un modello sulla GPU via steering server.
    Prima controlla e svuota la GPU, poi carica e verifica.
    Ritorna True se il modello è pronto.
    """
    # Controlla se è già caricato
    st = get_gpu_status(steering_url)
    if st.get("active_model") == model_name:
        log(f"  [gpu] {model_name} già attivo — niente da fare")
        return True

    # Unload se c'è qualcos'altro
    if st.get("active_model"):
        log(f"  [gpu] Modello diverso presente ({st['active_model']}) — unload prima")
        gpu_unload(steering_url, log=log)

    # Carica
    log(f"  [gpu] Caricamento {model_name}...")
    result = _post(f"{steering_url}/api/load_model", {"name": model_name})
    if not result:
        log(f"  [gpu] ERRORE: load_model non ha risposto")
        return False

    # Attendi che sia pronto
    deadline = time.time() + WAIT_LOAD_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        st = get_gpu_status(steering_url)
        if st.get("active_model") == model_name:
            log(f"  [gpu] {model_name} pronto — VRAM: {st.get('vram_used_gb', 0):.1f} GB")
            return True
        log(f"  [gpu] Attendo caricamento... (attivo: '{st.get('active_model', '')}')")

    log(f"  [gpu] TIMEOUT caricamento {model_name} dopo {WAIT_LOAD_TIMEOUT_S}s")
    return False


def gpu_prepare_for_probe(steering_url: str, log=print) -> bool:
    """
    Prepara la GPU per un probe diretto (probe_concept.py):
    scarica tutto e verifica VRAM libera.
    """
    log("  [gpu] Preparo GPU per probe diretto...")
    return gpu_unload(steering_url, log=log)


def gpu_restore_after_probe(steering_url: str, model_name: str, log=print) -> bool:
    """
    Dopo un probe diretto, ricarica il modello sullo steering server.
    """
    log(f"  [gpu] Ripristino {model_name} dopo probe...")
    return gpu_load(steering_url, model_name, log=log)
