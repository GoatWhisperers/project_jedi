#!/bin/bash
# setup_services.sh — Installa/aggiorna i systemd services per Project Jedi
#
# ESEGUIRE UNA VOLTA SOLA con sudo:
#   sudo bash project_jedi/scripts/setup_services.sh
#
# Cosa fa:
#   1. Aggiorna llama-server-m40.service → CUDA, build_cuda, 12B, --n-gpu-layers 99
#   2. Crea steering-server.service     → avvio automatico steering_server.py
#   3. Ricarica systemd e riavvia i servizi

set -e

echo "=== Project Jedi — Setup systemd services ==="
echo ""

# ── 1. llama-server-m40 ────────────────────────────────────────────────────────
echo "[1/3] Aggiornamento llama-server-m40.service..."
cat > /etc/systemd/system/llama-server-m40.service << 'SVCEOF'
[Unit]
Description=llama-server Gemma3-12B Q4_K_M - Project Jedi M40 GPU (CUDA)
After=network.target

[Service]
Type=simple
User=lele
Group=lele
Environment="LD_LIBRARY_PATH=/mnt/raid0/llama-cpp-m40/build_cuda/bin:/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu"
Environment="OMP_NUM_THREADS=8"
ExecStart=/mnt/raid0/llama-cpp-m40/build_cuda/bin/llama-server \
    --model /mnt/raid0/models-gguf/gemma-3-12b-it-Q4_K_M.gguf \
    --host 0.0.0.0 --port 11435 \
    --ctx-size 4096 \
    --parallel 1 \
    --threads 8 \
    --n-gpu-layers 99 \
    --log-disable
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llama-server-m40

[Install]
WantedBy=multi-user.target
SVCEOF
echo "  ✓ llama-server-m40.service aggiornato"

# ── 2. steering-server ─────────────────────────────────────────────────────────
echo "[2/3] Creazione steering-server.service..."
cat > /etc/systemd/system/steering-server.service << 'SVCEOF'
[Unit]
Description=Project Jedi Steering Server (MI50 ROCm, porta 8010)
After=network.target

[Service]
Type=simple
User=lele
Group=lele
WorkingDirectory=/home/lele/codex-openai
ExecStart=/home/lele/codex-openai/project_jedi/.venv/bin/python \
    /home/lele/codex-openai/project_jedi/scripts/steering_server.py
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=steering-server

[Install]
WantedBy=multi-user.target
SVCEOF
echo "  ✓ steering-server.service creato"

# ── 3. Reload e riavvio ────────────────────────────────────────────────────────
echo "[3/3] Reload systemd + riavvio servizi..."
systemctl daemon-reload

systemctl enable steering-server.service
echo "  ✓ steering-server abilitato"

# Riavvia M40 (ferma quello sbagliato e parte il nuovo)
systemctl restart llama-server-m40.service
echo "  ✓ llama-server-m40 riavviato"

# Avvia steering (se non già attivo)
systemctl restart steering-server.service
echo "  ✓ steering-server avviato"

echo ""
echo "=== Verifica stato ==="
sleep 5
systemctl is-active llama-server-m40.service && echo "  llama-server-m40: ACTIVE" || echo "  llama-server-m40: FAILED"
systemctl is-active steering-server.service   && echo "  steering-server:  ACTIVE" || echo "  steering-server:  FAILED"

echo ""
echo "Attendi ~30s poi verifica:"
echo "  curl http://localhost:11435/health"
echo "  curl http://localhost:8010/api/models"
echo "  nvidia-smi  (verifica VRAM M40 > 6 GB)"
