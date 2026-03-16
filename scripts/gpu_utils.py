"""
gpu_utils.py — Gestione sicura della GPU per Project Jedi.

Sequenza corretta per caricare un modello sulla GPU:
  1. Controlla cosa c'è sulla GPU e se sta ancora lavorando
  2. Aspetta che finisca (se busy=True)
  3. Unload pulito via mi50_manager (porta 8020)
  4. Verifica che la VRAM si sia svuotata
  5. Carica il nuovo modello
  6. Verifica che sia pronto

Uso:
    from gpu_utils import gpu_prepare, gpu_reload

    # Prima del probe (libera VRAM):
    gpu_prepare(steering_url="http://localhost:8020")

    # Dopo il probe (ricarica modello):
    gpu_reload(steering_url="http://localhost:8020", model_name="Gemma2-Uncensored")

NOTA: steering_url ora punta a mi50_manager (8020), non a steering_server (8010).
      decompose.py usa ancora --steering-url per compatibilità; il valore default
      è stato aggiornato in decompose.py. gpu_utils accetta qualsiasi URL.
"""

import json
import subprocess
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
    """
    Ritorna lo stato attuale della GPU leggendo da mi50_manager /api/status.
    Campi: active_model, busy, vram_used_gb, vram_total_gb.
    steering_url deve puntare a mi50_manager (default: http://localhost:8020).
    """
    st = _get(f"{steering_url}/api/status") or {}
    return {
        "active_model":  st.get("model", ""),
        "busy":          st.get("busy", False),
        "vram_used_gb":  st.get("vram_used_gb", 0),
        "vram_total_gb": st.get("vram_total_gb", 0),
    }


def wait_gpu_idle(steering_url: str, log=print) -> bool:
    """
    Aspetta che mi50_manager non stia elaborando (busy=False).
    Ritorna True se idle entro il timeout, False altrimenti.
    """
    deadline = time.time() + WAIT_BUSY_TIMEOUT_S
    while time.time() < deadline:
        st = get_gpu_status(steering_url)
        if not st.get("busy", False):
            return True
        log(f"  [gpu] GPU occupata (busy=True) — attendo...")
        time.sleep(5)
    log(f"  [gpu] TIMEOUT attesa GPU idle dopo {WAIT_BUSY_TIMEOUT_S}s")
    return False


def gpu_unload(steering_url: str, log=print) -> bool:
    """
    Sequenza completa di unload via mi50_manager:
      1. Legge stato attuale
      2. Aspetta che la GPU sia idle
      3. POST /api/unload_model
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

    # 2. Unload via POST a mi50_manager
    log("  [gpu] Unload modello...")
    result = _post(f"{steering_url}/api/unload_model", {})
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
    Carica un modello sulla GPU via mi50_manager.
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


# ── Verifica M40 su GPU (CUDA) ─────────────────────────────────────────────────

M40_VRAM_MIN_MB       = 1000   # soglia: meno di 1 GB → sicuramente su CPU
M40_EXPECTED_MODEL    = "gemma-3-12b"   # sottostringa attesa nel nome modello
M40_CORRECT_BINARY    = "build_cuda"    # deve essere nel path del processo

def check_m40_on_gpu(m40_url: str = "http://localhost:11435", log=print) -> bool:
    """
    Verifica che llama-server M40 stia girando sulla GPU CUDA con il modello corretto.

    Controlli:
      1. nvidia-smi: VRAM usata sul M40 > soglia (altrimenti CPU)
      2. /v1/models: modello caricato contiene '12b' nel nome
      3. ps aux: il processo usa il binario build_cuda (non build/)

    Ritorna True se tutto OK, False e logga l'errore altrimenti.
    Solleva RuntimeError se trova CPU invece di GPU (blocca il pipeline).
    """
    ok = True

    # 1. VRAM check via nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        vram_mb = int(out.split("\n")[0])
        if vram_mb < M40_VRAM_MIN_MB:
            log(f"  [m40] ERRORE: VRAM M40 usata solo {vram_mb} MB — llama-server è su CPU!")
            raise RuntimeError(
                f"M40 llama-server sta girando su CPU (VRAM={vram_mb} MB < {M40_VRAM_MIN_MB} MB).\n"
                f"Riavviare con: /mnt/raid0/llama-cpp-m40/start_cuda.sh"
            )
        log(f"  [m40] VRAM M40: {vram_mb} MB ✓")
    except FileNotFoundError:
        log("  [m40] WARN: nvidia-smi non trovato, salto verifica VRAM")

    # 2. Modello giusto
    info = _get(f"{m40_url}/v1/models")
    if info:
        models = info.get("data") or info.get("models", [])
        names = [m.get("id", "") or m.get("name", "") for m in models]
        if not any(M40_EXPECTED_MODEL in n.lower() for n in names):
            log(f"  [m40] WARN: modello caricato non è 12B — trovato: {names}")
            log(f"        Riavviare con: /mnt/raid0/llama-cpp-m40/start_cuda.sh")
            ok = False
        else:
            log(f"  [m40] Modello: {names[0]} ✓")

    # 3. Binario corretto
    try:
        ps = subprocess.check_output(
            ["ps", "aux"], stderr=subprocess.DEVNULL, text=True
        )
        llama_lines = [l for l in ps.splitlines() if "llama-server" in l and "grep" not in l]
        for line in llama_lines:
            if M40_CORRECT_BINARY not in line:
                log(f"  [m40] WARN: llama-server usa binario SBAGLIATO (manca '{M40_CORRECT_BINARY}'):")
                log(f"        {line.split()[10] if len(line.split()) > 10 else line[:120]}")
                log(f"        Riavviare con: /mnt/raid0/llama-cpp-m40/start_cuda.sh")
                ok = False
            else:
                log(f"  [m40] Binario: {M40_CORRECT_BINARY} ✓")
    except Exception as e:
        log(f"  [m40] WARN: impossibile verificare binario: {e}")

    return ok
