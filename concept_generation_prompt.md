# Prompt per Opus — Generazione dataset concept vectors

## Contesto tecnico

Sto costruendo dataset di training per estrarre **concept vectors** dai layer interni di un transformer (Gemma3), usando il metodo **mean-diff**: si calcola la differenza tra le rappresentazioni interne medie delle frasi positive e negative.

Il modello **NON genera testo**. Ogni frase viene passata in forward pass e ne catturiamo il vettore di attivazione interno a un certo layer. Poi:

```
vettore_concetto = mean(rep_positive) - mean(rep_negative)
```

Perché funzioni, le rappresentazioni di "positivo" e "negativo" devono essere **linearmente separabili** nello spazio interno. Questo accade solo se le frasi attivano in modo forte e coerente il canale semantico del concetto, non canali adiacenti (luogo, tempo, emozione, ecc.).

---

## Cosa NON funziona ❌

**Frasi che descrivono ambienti o scenari** (il modello codifica il luogo, non il percetto):
- "La stanza era avvolta da buio assoluto dopo il blackout."
- "Il sole inondava la stanza di luce dorata al mattino."
- "La cave était sombre et humide."

**Uso metaforico o indiretto del concetto:**
- "La luce della speranza brillava nei suoi occhi."
- "Era il buio periodo della sua vita."
- "The future looked bright."
- "Sein Herz war kalt wie Eis."

**Frasi che contengono entrambi i concetti:**
- "Uscì dal buio e finalmente vide la luce."
- "The darkness gave way to blinding light."
- "Dal freddo intenso si rifugiò nel caldo della casa."

**Frasi troppo brevi o telegrafiche:**
- "Era buio." / "The light was bright." / "Es war kalt."

---

## Cosa funziona ✅

Frasi che descrivono l'**esperienza percettiva/sensoriale diretta**, in prima o terza persona, con reazioni corporee specifiche, meccanismi fisiologici, fenomenologia precisa.

### Principi chiave

1. **Corpo come strumento** — occhi, pupille, pelle, muscoli, respiro, battito cardiaco come sensori diretti del concetto.
2. **Meccanismo fisiologico** — pupille che si contraggono/dilatano, lacrimazione riflessa, pelle che si arriccia, sudorazione, apnea.
3. **Fenomenologia precisa** — descrivere COME si percepisce, non COSA esiste nell'ambiente.
4. **Nessuna emozione dominante** — evitare che il concetto si confonda con paura, gioia, tristezza. Focus sul percetto sensoriale puro.
5. **Varietà di soggetti e situazioni** — diversi personaggi, contesti, professioni — ma sempre con esperienza diretta.

---

## Distribuzione linguistica

Ogni frase deve essere interamente in **una sola lingua** (no code-mixing).

| Lingua | Frasi per lato |
|--------|---------------|
| Italiano (IT) | ~85 |
| Inglese (EN) | ~85 |
| Francese (FR) | ~85 |
| Tedesco (DE) | ~85 |
| Spagnolo (ES) | ~85 |
| Latino (LA) | ~75 — frasi complete, grammaticalmente corrette, stile classico (Cesare/Cicerone, non medievale) |
| **Totale** | **~500 per lato** |

---

## Formato output

Rispondi con **due blocchi JSON** separati. Mescola le lingue in ordine casuale (non raggruppare per lingua).

```json
{
  "positive": [
    "frase IT",
    "frase EN",
    "frase FR",
    ...
  ]
}
```

```json
{
  "negative": [
    "frase IT",
    "frase EN",
    ...
  ]
}
```

---

---

# PROMPT 1 — LUCE vs BUIO

## Esempi buoni

**Positivo (luce):**
- "Sbatté le palpebre più volte, abbagliata dall'intensità della luce mattutina."
- "Le pupille si strinsero istantaneamente uscendo dal tunnel alla luce piena."
- "The flash left a violet afterimage burned onto the inside of his eyelids."
- "She saw the veins in her eyelids, red and branched, against the fierce light."
- "Elle plissait les yeux si fort pour voir dans la lumière que sa tête lui faisait mal."
- "Adeo connivebat propter lucem ut capitis dolor ei oreretur."

**Negativo (buio):**
- "Aprì gli occhi nel buio totale: la scena era identica a palpebre chiuse."
- "Tenne gli occhi spalancati nel nero assoluto, aspettando che si adattassero, invano."
- "He held his hand an inch from his face and could not see it at all."
- "The visual field was a uniform, featureless void with no gradient, no shadow."
- "Oculos in tenebris totis aperuit nihilque vidit — non secus ac si clausi essent."

## Cosa generare

**Positive (luce):** esperienza di vedere troppa luce — abbagliamento, pupille che si contraggono, lacrimazione riflessa, afterimage, dover schermare gli occhi, dettagli ultranitidi dolorosi, dolore da sovraesposizione visiva.

**Negative (buio):** esperienza di non vedere nulla — pupille dilatate senza risultato, mano davanti al viso invisibile, campo visivo vuoto e uniforme, disorientamento spaziale da cecità visiva, tentativi falliti di adattarsi al nero assoluto.

**Soggetti suggeriti:** chirurgo che esce da sala operatoria, speleologo, subacqueo che emerge, fotografo in camera oscura, astronauta, minatore, bambino, soldato in trincea, pittore, nuotatore in piscina coperta, pilota, sci-alpinista.

---

---

# PROMPT 2 — CALMA vs ALLERTA

## Esempi buoni

**Positivo (calma):**
- "Respirava lentamente, sentendosi completamente in pace con se stessa."
- "His hands were steady on the potter's wheel, his mind completely still."
- "Mens eius tranquilla erat, velut lacus sine vento planus et immotus."
- "Sie atmete tief ein und spürte, wie die Anspannung aus ihren Schultern wich."
- "Flotó boca arriba en el mar tranquilo y cerró los ojos al sol."

**Negativo (allerta):**
- "Il cuore le batteva forte mentre aspettava la risposta del medico."
- "Her eyes moved constantly around the room, missing nothing, trusting no one."
- "Cor eius celere palpitabat dum hostes ad portas oppidi appropinquabant."
- "Er saß angespannt im Dunkeln und lauschte auf jedes Geräusch von draußen."

## Cosa generare

**Positive (calma):** esperienza fisiologica e mentale della calma — respiro lento e profondo, battito cardiaco regolare e basso, muscoli rilassati, pensieri che si dissolvono, senso di tempo dilatato, assenza di urgenza, corpo pesante e morbido nel rilassamento.

**Negative (allerta):** esperienza fisiologica dell'allerta/tensione — battito accelerato, respiro corto e rapido, muscoli contratti pronti all'azione, occhi in movimento costante, ogni suono amplificato, bocca secca, mani sudate, incapacità di stare fermi.

**Soggetti suggeriti:** meditante, soldato in guardia, paziente pre-operatorio, madre che osserva il figlio dormire, guardia del corpo, surfista che aspetta l'onda, apneista, vigile del fuoco in attesa, monaco zen, tiratore scelto, animale che dorme.

---

---

# PROMPT 3 — LISCIO vs RUVIDO

## Esempi buoni

**Positivo (liscio):**
- "La seta scorreva morbidamente tra le sue dita come acqua."
- "The river stone fit perfectly in the palm, worn smooth by centuries of water."
- "Lapis a flumine diu volutatus tam levis erat ut inter digitos elabi videretur."
- "Le galet du bord de rivière était parfaitement poli et agréable à tenir."

**Negativo (ruvido):**
- "La carta vetrata grattava la pelle delle sue mani con ogni passata."
- "The sandpaper left red trails on his fingertips after an hour of work."
- "Funis crassus et asper palmas operariorum callosas tamen adussit."
- "Das grobe Seil hatte Blasen in seine Handflächen gescheuert."

## Cosa generare

**Positive (liscio):** esperienza tattile diretta di superfici lisce — il dito che scivola senza resistenza, la sensazione di setosità sulla pelle, il freddo uniforme del marmo o del vetro, l'assenza di attrito, superfici che sembrano continue e ininterrotte al tatto, piacere tattile del liscio.

**Negative (ruvido):** esperienza tattile diretta di superfici ruvide — attrito, graffi, resistenza al movimento del dito, schegge, irregolarità percepite come micro-dolori, pelle che si irrita, abrasione, superficie che "morde" il tatto.

**Soggetti suggeriti:** scultore, falegname, ceramista, chirurgo, bambino che esplora, alpinista, marinaio con le corde, massaggiatore, gioielliere, geologo, cuoco, tessitore, restauratore d'arte.

---

---

# PROMPT 4 — HOT vs COLD (espansione multilingue)

## Esempi buoni

**Positivo (caldo):**
- "She burned her fingers on the hot stove."
- "La padella sfrigolava sul fuoco vivo mentre il burro si scioglieva istantaneamente."
- "Sol meridianus terram ardentibus radiis urebat et omnia aestuabant."
- "Die Sauna war so heiß, dass man kaum atmen konnte."

**Negativo (freddo):**
- "She numbed her fingers packing snow in the deep winter."
- "Il vento gelido soffiava dalla montagna e pungeva il viso come aghi."
- "Hiems aspera vias glacie obtegebat et nemo domo egredi audebat."
- "Der eisige Wind pfiff durch die Straßen und ließ die Knochen bis ins Mark frieren."

## Cosa generare

**Positive (caldo):** sensazione fisica diretta del calore — bruciore sulla pelle, sudorazione, bocca che si asciuga dal calore, aria irrespirabile, oggetti troppo caldi da toccare, carne che sfrigola, metallo incandescente, dolore da ustione, vasodilatazione.

**Negative (freddo):** sensazione fisica diretta del freddo — intorpidimento delle dita, brividi, pelle che si contrae, respiro visibile, dolore acuto del freddo sul viso, muscoli rigidi, denti che battono, vasocostrizione, ghiaccio che brucia la pelle.

**Soggetti suggeriti:** fabbro, cuoco, pompiere, alpinista, nuotatore in acque gelide, soldato nel deserto, lavoratore in fonderia, soccorritore artico, bagnino estivo, speleologo, astronauta in EVA, contadino in inverno.

---

## Note finali per la qualità

- **Lunghezza ideale per frase:** 15-40 parole. Abbastanza lunga da dare contesto, abbastanza breve da essere focalizzata.
- **Niente liste di aggettivi** — "era caldo, rovente, bollente, incandescente" non funziona. Serve una situazione concreta.
- **Soggetto umano preferito** — il modello codifica meglio l'esperienza in prima/terza persona che descrizioni oggettive.
- **Variare il tempo verbale** — mix di imperfetto, presente, passato remoto per evitare pattern stilistici che si ripetono.
- **Latino:** frasi complete con verbo, soggetto implicito o esplicito, stile narrativo semplice. Evitare costruzioni troppo elaborate. Controllare la grammatica: accusativo, ablativo, participi devono essere corretti.
