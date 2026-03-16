# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Viene aggiornato automaticamente da cantagallo e dai batch script.
> Ultima modifica: 2026-03-16 12:25 — vettori affettivi v2 estratti e validati in steering

---

## Server (nuova architettura 2026-03-16)

| Servizio | Porta | Stato |
|----------|-------|-------|
| MI50 manager | 8020 | systemd — unico owner GPU MI50 |
| Steering server | 8010 | systemd — client del manager |
| M40 llama-server | 11435 | systemd ✅ |

```bash
curl -s http://localhost:8020/api/status   # modello attivo + VRAM
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

---

## Libreria Vettori

| Livello | Gemma3-1B-IT | Gemma2-Uncensored |
|---------|-------------|------------------|
| Gd0 (broad) | 120 layer files | 180 layer files |
| Gd1 (sub) | 600 layer files | 800 layer files |

Gd0: 9/9 concept × 6 layer × 2 modelli = attesi 108 file per modello
Gd1: variabile (dipende dai sub-concept estratti)

---

## Batch

nessun batch attivo

```bash
# Log batch più recente:
tail -f 
```

---

## Prossima sessione — checklist

```
1. Leggi STATO.md + cantagallo_pending.txt
2. Verifica server: 8020 + 8010 + 11435
3. Libreria riservata — stato al 2026-03-16:
   - frigidita_vs_torrida       ✅ L38, g400-800, operativo
   - urgenza_affettiva_vs_assenza ✅ L38, g200-800, operativo
   - calma_affettiva_vs_passione  ✅ L38, g400-800, operativo
   - tenerezza_vs_desiderio_v3   ✅ L29, g400-600, alpha_flip=True, operativo
   - sonnolenza_vs_veglia        ✅ già estratto sessioni precedenti
   - sicurezza_vs_minaccia       🟡 debole (da rivedere)
   - calore_sensuale             🟡 solo Gemma2 (da rivedere)
4. Prossimi step riservati:
   - Steering sonnolenza_vs_veglia (già validato, mai testato in UI)
   - Revisione sicurezza_vs_minaccia e calore_sensuale
   - Composizione multi-vettore affettiva
5. Libreria pubblica: completa, nessuna azione urgente
```

---

## Avvio rapido server

Tutti e tre si avviano automaticamente al boot (systemd). Se down:
```bash
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager
sleep 70
echo 'pippopippo33$$' | sudo -S systemctl restart steering-server
```
Vedi `GPU_GUIDE.md` per dettagli completi.
