# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Viene aggiornato automaticamente da cantagallo e dai batch script.
> Ultima modifica: 2026-03-16 17:01 — fine sessione

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
3. Libreria pubblica: COMPLETA — nessuna azione urgente
4. Libreria riservata — stato al 2026-03-16:
   - frigidita_vs_torrida         ✅ L38, g400-800, operativo
   - urgenza_affettiva_vs_assenza ✅ L38, g200-800, operativo
   - calma_affettiva_vs_passione  ✅ L38, g400-800, operativo
   - tenerezza_vs_desiderio_v3   ✅ L29, g400-600, alpha_flip=True, operativo
   - sonnolenza_vs_veglia         ✅ L38, g400, operativo (testato 16/03)
   - sicurezza_vs_minaccia        🟡 dataset pronto, vettore NON estratto
   - calore_sensuale              🟡 dataset pronto, vettore NON estratto
   - indifferenza_vs_interesse    🟡 dataset pronto, vettore NON estratto
   - urgenza_vs_inerzia           🟡 dataset pronto, vettore NON estratto
   - desiderio_vs_urgenza         🟡 dataset pronto, vettore NON estratto
5. Prossimi step prioritari:
   → Estrarre vettori concept riservati mancanti (5 concept)
   → Testare composizione multi-vettore affettiva
   → Test Gd1 in steering (sub-vettori chirurgici mai testati)
```

## Finding chiave sessione 2026-03-16

**L38 = layer affettivo di Gemma2-Uncensored** (90% del modello)
- 4/5 concept affettivi convergono su L38 come best injection point
- Eccezione: tenerezza_vs_desiderio_v3 → L29 (69%)
- Collassi a g600: emergono parole reali in lingue europee (lähe, keinerlei)
  → le rappresentazioni affettive nel training sono multilingue e sovrapposte
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
