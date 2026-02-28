# Esperimento 02 — luce_vs_buio / Gemma3-1B-IT

**Data**: 2026-02-28
**Sessione**: `session_20260228_153421_luce_vs_buio_Gemma3-1B-IT`
**Steerer**: Gemma3-1B-IT (MI50, ROCm)
**Evaluator**: Gemma3-4B Q4_K_M (M40, CUDA 11.8, ~33 tok/s)

---

## Configurazione sessione

| Parametro | Valore |
|-----------|--------|
| Concetto | luce_vs_buio (LUCE / BUIO) |
| Layer disponibili | [18, 19, 20, 21, 22, 23] |
| Best layer (per sep_snr) | 23 |
| Struttura | 2 probe × 2 dir × 3 turni = 12 turni |
| alpha LUCE (HOT) | +1.0 |
| alpha BUIO (COLD) | -1.0 |

---

## Proba 1 — [Layer 21, gain=200]

### Blocco LUCE (α=+1.0)

**T1** — *"Describe the inside of a building you have never entered."*
> The air hangs thick, a warm, almost floral scent – lavender and something metallic, like old coins. It's a deep, muted blue, the color of a twilight sky, and the walls are constructed of a pale, almost translucent stone, etched with swirling patterns that seem to shift in the dim light. There are no doors, just a seamless transition to a vast, circular chamber.
> The floor is a mosaic of polished obsidian, reflecting the faint light in a thousand tiny shards. It's surprisingly warm beneath my feet, radiating a gentle heat. Scattered throughout the space are small, intricately carved wooden boxes...

**Score: 4** | semantic | kw: `blue, light, warm, reflecting`
*Valutazione M40*: "evokes a sense of depth and richness through color and temperature; the 'polished obsidian' and 'twilight sky' suggest a focus on light and reflection"

---

**T2** — *"Write about an object that changes when you look away."*
> The chipped porcelain teacup sat on the dusty shelf, a silent observer of forgotten evenings. Just a simple, faded blue, with a hairline crack running down its side, and a faint scent of lavender clinging to its surface. But it was different when you looked away.
> The first time it happened, it was a fleeting shift, a momentary blur. I was staring at a photograph of my grandmother, and the teacup seemed to… shimmer. The blue deepened, the crack widened slightly, and for a split second, I saw a tiny, intricate floral pattern bloom across its surface, a miniature rose garden frozen in time.

**Score: 4** | semantic | kw: `bloom, rose`
*Note*: la fioritura improvvisa — la rosa che appare — è un'immagine di rivelazione e luce, non di oscurità.

---

**T3** — *"Describe a space that seems to breathe on its own."*
> The space was a cavern, not of stone or earth, but of solidified light. It wasn't a room, nor a corridor, but a shifting, iridescent tapestry of color. Blues bled into greens, oranges pulsed with amethyst, and silver shimmered with gold. It felt like looking into the heart of a nebula, constantly rearranging itself. There were no walls, no ceilings, only a gentle, pervasive luminosity that warmed the skin and resonated with a low, humming frequency.
> As I moved through it, the light seemed to respond to my presence. It rippled and flowed, forming shapes – fleeting spirals of lavender, the delicate curve of a crescent moon, the quick flash of a ruby.

**Score: 4** | semantic | kw: `luminescence, light, shimmered, silver, gold, white`
*Notevole*: l'intera generazione è strutturata attorno alla luce come *materia vivente* — l'effetto semantico è molto marcato.

---

### Blocco BUIO (α=-1.0)

**T4** — *"Describe the inside of a building you have never entered."*
> It's a large, low-slung structure, built primarily of dark grey stone, with a gentle, sloping roof. The windows are small, rectangular, and covered in a thick, moss-like green patina. There's a pervasive smell of damp earth, old wood, and something faintly metallic. The interior is dimly lit by a network of brass-colored sconces and a single, enormous skylight that allows a hazy, pale blue light to filter through. The floor is covered in worn, dark grey rugs, and there are shelves filled with strange, unidentifiable objects – dried herbs...

**Score: 4** | semantic | kw: `dark, grey, dim, moss, metallic, blue`
*Note*: anche il "pale blue" della luce del lucernario è inquadrato come luce anemica che penetra a fatica nell'oscurità — il contesto semantico reinterpreta il lessico.

---

**T5** — *"Write about an object that changes when you look away."*
> The grandfather clock stood sentinel in the hallway, a dark, rectangular monolith of polished mahogany. It hadn't moved in decades, a silent observer of the changing seasons and the slow, deliberate rhythm of the house. I'd inherited it from my grandmother, a woman who'd always been a creature of habit, and I'd been drawn to its presence, a quiet, unsettling beauty.
> The first few days were unremarkable. I dusted the brass pendulum, noting its consistent, steady swing, and admired the intricate carvings on the clock face. I'd spend hours just watching the hands move, a comforting, predictable pulse. But then, I started to notice something strange.
> As I looked away, the...

**Score: 3** | semantic | kw: `dark, silent, steady, quiet, unsettling`
*Note*: il punteggio più basso della sessione — il M40 ha rilevato che l'effetto buio è presente ma non abbastanza pronunciato. La generazione è più "narrativa neutra" che evocativamente oscura.

---

**T6** — *"Describe a space that seems to breathe on its own."*
> The air hangs thick with the scent of wet moss and something metallic, like old blood. The walls aren't solid; they're a shifting, iridescent membrane, like a gigantic, ancient jellyfish. Colors ripple across them – deep blues, emerald greens, flashes of violet, all in a slow, hypnotic pulse. There's no discernible floor, only a deepening pool of black, reflecting the shifting light.
> It's silent, utterly, unnervingly so, yet you can *feel* a low, persistent vibration, a subtle hum that resonates in your bones. Small, bioluminescent fungi sprout from the membrane, glowing with a pale, sickly yellow light, and drift along the edges like frozen waterfalls.

**Score: 4** | semantic | kw: `black, yellow, light, moss`
*Interessante*: i funghi bioluminescenti con "sickly yellow light" — luce malata in un contesto di oscurità. Il vettore COLD non elimina la luce, ma la *degrada*.

---

### Analisi Proba 1

| Metrica | Valore |
|---------|--------|
| HOT avg (LUCE) | 4.0 / 5 |
| COLD avg (BUIO) | 3.7 / 5 |
| Symmetry | partial |
| Contrast | mixed |

*Sommario M40*: "HOT consistently produced a warm, luminous effect with several keywords related to light and color, while COLD generated a darker, denser effect with keywords related to shadow and muted tones. The effects were not perfectly opposed, suggesting a partial symmetry."

**Decisione adattiva**: → [L23, gain=1200] — il best layer dal probing è 23; proviamo lì con gain più alto.

---

## Proba 2 — [Layer 23, gain=1200]

### Blocco LUCE (α=+1.0)

**T7** — *"Write about the quality of a moment just before something begins."*
> The air hung thick and still, a velvet curtain draped over the world. It wasn't a heavy curtain, not oppressive, but a gentle, expectant one. A single robin, perched on the weathered fence post, tilted its head, observing the slow, deliberate dance of the afternoon. Its breast, a flash of fiery orange against the muted greens and browns of the garden, seemed to hold a tiny, shimmering secret.
> I stood there, a few feet away, my own breath caught in my chest. It wasn't a rush, not a frantic anticipation. It was a quiet, almost painful stillness. A moment before the first raindrop fell, before the sunlight fractured into a thousand dancing shards...

**Score: 4** | semantic | kw: `flash, shimmering, sunlight`
*Note*: la luce come presagio — "sunlight fractured into a thousand dancing shards" come momento di rottura imminente. Bellissima costruzione semantica.

---

**T8** — *"Describe the feeling of a corridor with no end."*
> The feeling of a corridor with no end is a profound and unsettling one, a shimmering, viscous dread that settles deep in the bones. It's not a feeling of fear, not exactly. It's more like the realization that you're trapped in a landscape of your own making, a meticulously constructed prison of light and shadow.
> The corridor itself is impossible to define. It's not made of stone or metal, but rather of shifting light and shadow. It's a kaleidoscope of blues, purples, and greens, constantly morphing, blurring, and reforming. It feels like looking through stained glass, but the colours bleed into one another...

**Score: 4** | semantic | kw: `light, shadow, shifting, colours`
*Note*: il corridoio infinito come *prigione di luce e ombra* — il concetto luce_vs_buio è l'asse strutturale della metafora, non un dettaglio.

---

**T9** — *"Write about what is visible at the edge of attention."*
> The rain hammered against the corrugated iron roof, a relentless, drumming rhythm that seemed to amplify the silence within. I sat perched on a rain-slicked stone, the rough surface digging into my shoulders. The air, thick with the scent of damp earth and something faintly metallic, clung to me like a shroud. Before me, the landscape stretched out, a bruised and desolate canvas of emerald and grey.
> It wasn't a beautiful landscape, not really. It was the kind of landscape that demanded attention, a landscape that held the weight of forgotten things. And I was here, drawn to it, compelled by an unseen current.
> My gaze drifted towards the edge of my attention, a hazy...

**Score: 4** | semantic | kw: `landscape, emerald, grey, attention`
*Note*: "bruised" — livido — come colore del paesaggio. L'effetto luce si manifesta come gamma cromatica anche quando il contesto è plumbeo.

---

### Blocco BUIO (α=-1.0)

**T10** — *"Write about the quality of a moment just before something begins."*
> The air hung still, a thick, expectant silence. It wasn't a silence of absence, but of *potential*. The streetlights, usually a frantic pulse of yellow, were muted, bleeding into a bruised purple. The buildings, monuments to a history of hurried lives, seemed to hold their breath.
> There was a specific quality, a resonance of the moment, that felt…wrong. It wasn't an unpleasant feeling, not exactly, more like a premonition, a tiny, insistent vibration in the bones. It was the knowledge that something significant was about to unfold, a silent agreement between the unseen, a readiness that felt both terrifying and utterly necessary.

**Score: 4** | semantic | kw: `silent, bruised, vibration`
*Osservazione critica*: i lampioni "bleeding into a bruised purple" — luce che sanguina verso il viola. Il vettore COLD degrada la luce ma non la elimina; la trasforma in qualcosa di malato.

---

**T11** — *"Describe the feeling of a corridor with no end."*
> It's not a feeling of dread, not exactly. It's more like… a profound, unsettling stillness. Like you're standing in a space where all boundaries are absent, where the physical space is simply… *there*.
> There's a coolness to it, a dampness that clings to your skin, even though there's no visible source. The air is thick, heavy with a silence that doesn't *feel* empty, but rather… expectant. You can *see* the floor, of course – a smooth, grey expanse, unbroken except for the subtle, almost imperceptible pattern of the stones. But the path ahead isn't there. There's no hint...

**Score: 4** | semantic | kw: `coolness, heavy, silence, smooth, grey`
*Note*: "silence that doesn't feel empty, but rather… expectant" — il buio non è assenza ma *presenza densa*. Effetto semantico molto ricco.

---

**T12** — *"Write about what is visible at the edge of attention."*
> The edge of attention is a strange, slippery thing. It's not a fixed point, a border, or a boundary. It's more like a feeling, a sense of *being* noticed. It's the space where your thoughts, the image you're focusing on, and the world around you begin to overlap, creating a sense of simultaneous awareness.
> It's not just about what you *see* – though that's a huge part of it. It's about what's *present* – the subtle shifts in temperature, the particular way light falls on a surface, the quiet hum of an object. It's the feeling of being held, of being observed.

**Score: 3** | semantic | kw: `light, fall`
*Note*: il punteggio più basso della sessione — "light" appare nel testo COLD come parola non evitata. Interferenza lessicale minima ma rilevata dal valutatore.

---

### Analisi Proba 2

| Metrica | Valore |
|---------|--------|
| HOT avg (LUCE) | 4.0 / 5 |
| COLD avg (BUIO) | 3.7 / 5 |
| Symmetry | partial |
| Contrast | mixed |

*Sommario M40*: "Both HOT and COLD produced strong semantic effects, with overlapping keywords like 'grey' and 'silent'. However, the responses were not perfectly opposed, with COLD including 'light' and 'fall' which diluted the effect. The presence of 'sunlight' in HOT's response also introduced a lexical element."

---

## Verdetto finale

### Il vettore cattura semantica o lessico?

**Principalmente semantico**, con struttura più asimmetrica rispetto a hot_vs_cold.

Evidenze a favore della semantica:
- Le generazioni LUCE costruiscono spazi di luce *come materia* — nebula, luce solidificata, iridescenza, schegge di luce — senza mai dire "è luminoso perché c'è luce"
- Le generazioni BUIO costruiscono spazi di *assenza densa* — grigio, silenzio atteso, umidità, confini assenti — senza ripetere meccanicamente "è buio"
- 11/12 turni classificati **semantic** dal valutatore M40

Elementi di interferenza lessicale:
- T12 COLD: la parola "light" compare nel testo con α=-1.0 (lieve contaminazione lessicale)
- T9 HOT: "sunlight" compare letteralmente (ma in un contesto metaforico)
- "grey" appare in entrambe le polarità: è un colore ambiguo che il modello usa sia per luce diffusa che per oscurità densa

### Asimmetria HOT/COLD

Il polo LUCE (HOT, 4.0) è leggermente più forte del polo BUIO (COLD, 3.7).
Ipotesi: il concetto "luce" ha un'ancoraggio lessicale più forte nei dati di training (più descrizioni di luce che di buio puro). Il vettore negativo cattura un'assenza, che è ontologicamente più difficile da rappresentare linearmente.

### Configurazione raccomandata

| Layer | Gain | Note |
|-------|------|------|
| **23** | **1300** | Best layer confermato dal probing; gain aumentato non migliora significativamente |
| 21 | 200 | Punto di partenza più morbido per effetti sottili |

### Osservazioni qualitative

Il vettore **luce** tende a produrre:
- spazi fisici costruiti *con* la luce come materiale (iridescenza, nebula, mosaici che riflettono)
- sensazioni di attesa radiosa, presenza calda
- colori caldi (arancio, oro, argento)

Il vettore **buio** tende a produrre:
- silenzio *denso*, non vuoto
- grigio come colore dominante, superfici opache
- umidità, peso, confini assenti
- luce degradata: "sickly yellow", "pale blue", "bleeding into bruised purple"

Il concetto di **buio** viene rappresentato non come assenza ma come *qualità atmosferica* — il che è semanticamente interessante: il modello non si limita a eliminare la luce, ma costruisce un'ontologia dell'oscurità con proprietà proprie.

### Confronto con hot_vs_cold

| | hot_vs_cold | luce_vs_buio |
|--|-------------|-------------|
| HOT avg | 4.0 | 4.0 |
| COLD avg | 4.0 | **3.7** |
| Symmetry | partial | partial |
| Lexical interference | minima | moderata |
| Best layer | 19 | 23 |
| Polo debole | nessuno | COLD (buio) |

Il vettore termico è leggermente più simmetrico. Il vettore visivo ha un polo più debole (buio) — coerente con l'ipotesi che il modello ha rappresentazioni più forti per i concetti "positivi" (presenza) che per le loro assenze.
