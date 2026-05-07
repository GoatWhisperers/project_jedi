# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-05-07 — inizio integrazione Gemma4

---

## Server (architettura 2026-03-16, aggiornato 2026-05-07)

| Servizio | Porta | Stato |
|----------|-------|-------|
| MI50 manager | 8020 | systemd ✅ |
| Steering server | 8010 | systemd |
| M40 llama-server | 11435 | systemd (Gemma3-12B GGUF) |

```bash
curl -s http://localhost:8020/api/status   # modello attivo + VRAM
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

---

## Modelli disponibili

| Nome | Path | Layer | Hidden | Stato |
|------|------|-------|--------|-------|
| Gemma2-Uncensored | /mnt/raid0/gemma-2-uncensored | 42 | 3584 | ✅ vettori completi |
| Gemma3-4B-IT | /mnt/raid0/gemma-3-4b-it | 34 | 2560 | 🟡 nessuna estrazione |
| Gemma4-E4B-IT | /mnt/raid0/gemma-4-E4B-it | 42 | 2560 | 🟡 nessuna estrazione |
| ~~Gemma3-1B-IT~~ | ~~rimosso dal disco~~ | 26 | 1152 | 📦 vettori conservati |

**NOTA**: Gemma3/4 usano architettura multimodale (Gemma3/4ForConditionalGeneration).
mi50_manager aggiornato per gestirle. transformers aggiornato a 5.5.0.

---

## Libreria Vettori

| Livello | Gemma3-1B-IT | Gemma2-Uncensored | Gemma3-4B-IT | Gemma4-E4B-IT |
|---------|-------------|------------------|-------------|---------------|
| Gd0 (broad) | 📦 120 file (modello rimosso) | ✅ 180 file | 🔴 nessuno | 🔴 nessuno |
| Gd1 (sub) | 📦 ~600 file (modello rimosso) | ✅ ~800 file | 🔴 nessuno | 🔴 nessuno |

Archivio ricerche precedenti: `experiments/09_archivio_ricerche_precedenti.md`

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
3. AZIONE URGENTE: estrarre Gd0 per tutti i 9 concept su Gemma4-E4B-IT
4. Poi: estrarre Gd0 su Gemma3-4B-IT
5. Poi: concept affettivi riservati mancanti su Gemma2-Uncensored
   - sicurezza_vs_minaccia, calore_sensuale, indifferenza_vs_interesse
   - urgenza_vs_inerzia, desiderio_vs_urgenza
```

## Finding chiave sessione 2026-03-16

**L38 = layer affettivo di Gemma2-Uncensored** (90% del modello)
- 4/5 concept affettivi convergono su L38 come best injection point
- Sweet spot universale: **g400, alpha ±1.0, system prompt sensoriale obbligatorio**

---

## Avvio rapido server

Tutti e tre si avviano automaticamente al boot (systemd). Se down:
```bash
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager
sleep 70
echo 'pippopippo33$$' | sudo -S systemctl restart steering-server
```
Vedi `GPU_GUIDE.md` per dettagli completi.
- usa la memoria condivisa in /mnt/raid0/memoria_ai
