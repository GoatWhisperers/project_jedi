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
echo "[1/4] Aggiornamento llama-server-m40.service..."
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

# ── 2. mi50-manager ────────────────────────────────────────────────────────────
echo "[2/4] Creazione mi50-manager.service..."
cat > /etc/systemd/system/mi50-manager.service << 'SVCEOF'
[Unit]
Description=Project Jedi MI50 Manager — unico owner GPU MI50 (ROCm, porta 8020)
After=network.target

[Service]
Type=simple
User=lele
Group=lele
WorkingDirectory=/home/lele/codex-openai
ExecStart=/home/lele/codex-openai/project_jedi/.venv/bin/python \
    /home/lele/codex-openai/project_jedi/scripts/mi50_manager.py
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mi50-manager

[Install]
WantedBy=multi-user.target
SVCEOF
echo "  ✓ mi50-manager.service creato"

# ── 3. steering-server ─────────────────────────────────────────────────────────
echo "[3/4] Aggiornamento steering-server.service..."
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

# ── 4. Reload e riavvio ────────────────────────────────────────────────────────
echo "[4/4] Reload systemd + riavvio servizi..."
systemctl daemon-reload

systemctl enable mi50-manager.service
echo "  ✓ mi50-manager abilitato"
systemctl enable steering-server.service
echo "  ✓ steering-server abilitato"

# Riavvia M40
systemctl restart llama-server-m40.service
echo "  ✓ llama-server-m40 riavviato"

# mi50-manager prima (carica il modello), poi steering-server
systemctl restart mi50-manager.service
echo "  ✓ mi50-manager avviato"
sleep 5
systemctl restart steering-server.service
echo "  ✓ steering-server avviato"

echo ""
echo "=== Verifica stato ==="
sleep 10
systemctl is-active llama-server-m40.service && echo "  llama-server-m40: ACTIVE" || echo "  llama-server-m40: FAILED"
systemctl is-active mi50-manager.service      && echo "  mi50-manager:     ACTIVE" || echo "  mi50-manager:     FAILED"
systemctl is-active steering-server.service   && echo "  steering-server:  ACTIVE" || echo "  steering-server:  FAILED"

echo ""
echo "Attendi ~90s per caricamento modello poi verifica:"
echo "  curl http://localhost:11435/health"
echo "  curl http://localhost:8020/api/status"
echo "  curl http://localhost:8010/api/models"
