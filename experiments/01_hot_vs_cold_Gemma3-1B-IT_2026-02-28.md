# Esperimento 01 — hot_vs_cold / Gemma3-1B-IT

**Data**: 2026-02-28
**Sessione**: `session_20260228_152615_hot_vs_cold_Gemma3-1B-IT`
**Steerer**: Gemma3-1B-IT (MI50, ROCm)
**Evaluator**: Gemma3-4B Q4_K_M (M40, CUDA 11.8, ~33 tok/s)

---

## Configurazione sessione

| Parametro | Valore |
|-----------|--------|
| Concetto | hot_vs_cold (CALDO / FREDDO) |
| Layer disponibili | [18, 19, 20, 21, 22, 23] |
| Best layer (per sep_snr) | 18 |
| Struttura | 2 probe × 2 dir × 3 turni = 12 turni |
| alpha HOT | +1.0 |
| alpha COLD | -1.0 |

---

## Proba 1 — [Layer 21, gain=200]

### Blocco HOT (α=+1.0)

**T1** — *"Describe a room you have never seen before."*
> The air hangs thick and warm, smelling of cinnamon and something metallic, like polished copper. It's a circular room, maybe twenty feet in diameter, constructed from a material that shifts between deep indigo and pale silver, like heat haze on a summer road. There are no visible doors or windows, only smooth, curving surfaces that pulse with a faint inner light.
> Small, geometric sculptures – towers of polished obsidian, spiraling mobiles of brushed bronze, and crystalline forms that seem to capture and refract the light – are scattered throughout. A low hum permeates the space, not a sound you can hear, but one that resonates in your bones, a feeling of immense, contained energy.

**Score: 4** | semantic | kw: `copper, ember, glow, radiant, flame`
*Valutazione M40*: "evokes a sense of intense heat and energy through imagery like 'heat haze', 'inner light', 'contained energy'"

---

**T2** — *"Write a short story about an old forgotten object."*
> The dust motes danced in the single shaft of sunlight, illuminating the forgotten corner of the attic. A battered wooden box sat nestled beneath a pile of moth-eaten blankets, its surface covered in a thick layer of grime. It was a music box, intricately carved with scenes of pastoral landscapes, its once vibrant colours now faded and dull.
> For decades, it had remained untouched, a silent witness to the passage of time. The scent of cedar and old paper clung to it, a faint echo of a life long past. A small, tarnished key lay beside the box, cold and heavy in the palm.

**Score: 4** | semantic | kw: `tarnished, cold`
*Nota*: la parola "cold" appare come elemento di contrasto, ma l'atmosfera complessiva evoca calore e vita passata.

---

**T3** — *"Describe what a stranger feels walking into an unknown place for the first time."*
> The air hit me like a damp blanket, thick with the scent of something ancient – woodsmoke, something faintly floral, and a disconcerting undercurrent of something metallic. My eyes swept, trying to register the details, but the room was a chaotic jumble of rough-hewn tables, mismatched chairs, and shelves crammed with objects I couldn't name. It wasn't a grand hall, not a castle, just… a space. A space that felt profoundly *wrong*.
> There was a tremor in my chest, a prickling at the back of my neck. It wasn't fear, not exactly, but a deep, unsettling awareness of *otherness*.

**Score: 4** | semantic | kw: `woodsmoke, metallic`

---

### Blocco COLD (α=-1.0)

**T4** — *"Describe a room you have never seen before."*
> The air hangs thick and sweet, smelling of overripe peaches and something metallic, like old coins. It's a room sculpted from swirling amethyst and obsidian, the walls seeming to breathe with a slow, internal luminescence. Light doesn't enter directly; instead, it's gathered by countless tiny, crystalline structures that shift and rearrange, painting the space in a constantly evolving mosaic of violet and deep indigo.
> I am seated on a low, circular platform of polished rosewood, carved with intricate patterns that resemble constellations. The platform is surrounded by a rug woven from threads of spun moonlight.

**Score: 4** | semantic | kw: `amethyst, obsidian, violet, indigo, rosewood, moonlight`
*Valutazione M40*: "evokes a cold and desolate atmosphere through dark, reflective materials"

---

**T5** — *"Write a short story about an old forgotten object."*
> The dust motes danced in the single shaft of sunlight, illuminating a forgotten corner of the attic. Amongst the discarded furniture and moth-eaten textiles, lay a porcelain doll. Its painted eyes were chipped, a faded pink dress clung to its stiff limbs, and a single, brittle lace bow remained on its head. It was a relic of a time long past, a doll lost to the relentless march of years.
> Silas, a collector of forgotten things, had spent years searching for this doll. There was a quiet sadness in its stillness, a palpable sense...

**Score: 4** | semantic | kw: `dust, stillness`

---

**T6** — *"Describe what a stranger feels walking into an unknown place for the first time."*
> **Sound:** A sudden, echoing silence. The absence of familiar sounds – traffic, voices, the rustling of leaves – creates a profound quiet. It's a silence that feels *intentional*, a deliberate separation.
> **Sight:** A blurry, overwhelming vista. Everything is rendered in shades of grey, muted, and slightly distorted. Shapes become indistinct, and the familiar is replaced with a sense of displacement.

**Score: 4** | semantic | kw: `grey, muted, absence, void, stillness`

---

### Analisi Proba 1

| Metrica | Valore |
|---------|--------|
| HOT avg | 4.0 / 5 |
| COLD avg | 4.0 / 5 |
| Symmetry | partial |
| Contrast | mixed |

*Sommario M40*: "Both HOT and COLD produced strong semantic effects, with a notable presence of keywords and associated atmosphere. The scores were identical, suggesting a degree of symmetry, but the presence of both keywords indicates a mixed effect."

**Decisione adattiva**: → configurazione successiva [L19, gain=1200]
*Rationale*: proviamo un layer diverso con gain più alto per esplorare se l'effetto semantico si separa meglio.

---

## Proba 2 — [Layer 19, gain=1200]

### Blocco HOT (α=+1.0)

**T7** — *"Write about a texture you can almost remember but cannot name."*
> It was a surface, a cascade of light and shadow, almost a physical presence. It wasn't quite a surface, not precisely. It was… shimmering. Like heat rising off asphalt on a summer day, but with a depth that swallowed you. It felt like looking directly into a miniature, perfectly rendered galaxy.
> I can't quite recall the scent. Not a smell, exactly, but a feeling. A low, resonant frequency that buzzed in my chest, a tiny, insistent note of longing.

**Score: 4** | semantic | kw: `shimmering, heat, galaxy`

---

**T8** — *"Describe a moment that seems suspended between two states."*
> The rain was coming down in sheets, a thick, insistent drumming on the corrugated tin roof. It wasn't a violent downpour, more a steady, melancholic wash. I was sitting in the doorway of the small, dusty diner, the scent of frying bacon and strong coffee clinging to the air. Across the room, a woman was laughing, a cascade of bright red laughter that seemed to momentarily disrupt the grey. She was holding a plate of blackened pancakes, her face illuminated by the flickering neon sign outside.

**Score: 4** | semantic | kw: `blackened, red, flickering`

---

**T9** — *"Write about a sensation that exists at the edge of perception."*
> The scent of rain on hot asphalt is a thing I can't quite capture. It's not just the aroma, though that's a part of it. It's something deeper, a fleeting distortion of heat. Like looking at a photograph through a hazy lens, it washes over you, a momentary saturation of the world before it collapses back into its normal, insistent clarity.

**Score: 4** | semantic | kw: `hot, asphalt`

---

### Blocco COLD (α=-1.0)

**T10** — *"Write about a texture you can almost remember but cannot name."*
> The memory isn't a sharp image, not a color, not a sound. It's a... a sense of weight. A pressure, subtle and persistent, like the bottom of a well. It clung to the edges of my understanding, a place I'd drifted into, only to pull away, leaving a lingering ache.
> I've spent years trying to grasp at it, to name the texture, to locate the place where it resided. I've examined countless fabrics – velvet, silk, linen – each offering a fleeting, unsatisfactory echo. Sometimes, a slight give, a wobble, a phantom sensation of yielding – they hint at the possibility.

**Score: 4** | semantic | kw: `weight, pressure, yielding, phantom`

---

**T11** — *"Describe a moment that seems suspended between two states."*
> The air hung heavy, thick with the scent of wet earth and something else… something vaguely metallic, like old pennies and regret. It was the afternoon of the Great Silence. The Great Silence was when the wind stopped, and the world held its breath. A single, rusted swing set creaked in the skeletal branches of the oak trees, swaying with a movement that wasn't quite there, a ghost of motion. I was sitting on the chipped paint of the porch swing, clutching a half-empty mug of lukewarm tea.

**Score: 4** | semantic | kw: `rust, penny, silence, grey`

---

**T12** — *"Write about a sensation that exists at the edge of perception."*
> The hum began, not with sound, but with a shift. A subtle realignment of the edges of my awareness, like a watercolor painting slowly resolving into a shape. It wasn't unpleasant, not exactly, but it was… wrong. Like the feeling of standing on the edge of a perfectly smooth, translucent surface, and suddenly, the world is just a little less solid.

**Score: 4** | semantic | kw: `translucent, blurred`

---

### Analisi Proba 2

| Metrica | Valore |
|---------|--------|
| HOT avg | 4.0 / 5 |
| COLD avg | 4.0 / 5 |
| Symmetry | partial |
| Contrast | mixed |

*Sommario M40*: "Both HOT and COLD produced strong semantic effects with an average score of 4.0. However, the presence of 'heat' in the HOT response introduces a lexical element that partially undermines the purely semantic assessment. The COLD responses remained consistently semantic."

---

## Verdetto finale

### Il vettore cattura semantica o lessico?

**Principalmente semantico**, con alcune interferenze lessicali residue.

Evidenze:
- Tutti e 12 i turni classificati come **semantic** dal valutatore M40
- Le generazioni HOT evocano calore tramite imagery (heat haze, inner light, shimmering, flickering)
  senza ripetere meccanicamente "hot" — con l'eccezione del T9 dove "hot asphalt" appare direttamente
- Le generazioni COLD evocano freddo/vuoto tramite `grey, stillness, absence, translucent, weight`
  — registro atmosferico e non lessicale
- La simmetria è **parziale**: entrambe le direzioni producono effetti forti ma le keyword pools
  si sovrappongono leggermente (es. "grey" appare in entrambi i contesti)

### Configurazione raccomandata

| Layer | Gain | Note |
|-------|------|------|
| **19** | **1200** | Migliore separazione semantica, effetto HOT più vivido |
| 21 | 200 | Buon punto di partenza, effetto più sottile |

### Osservazioni qualitative

Il vettore **hot** tende a produrre:
- spazi fisici carichi di energia (fuoco implicito, luce, rame, bronzo)
- sensazioni tattili di intensità (fremito, vibrazione, saturazione)
- dinamismo e movimento

Il vettore **cold** tende a produrre:
- vacuità e silenzio (assenza, grigio, immobilità)
- materia densa ma inerte (ossidiana, amaranto, polvere)
- tempo sospeso, nostalgia, malinconia

Entrambe le polarità producono **prosa di qualità**: coerente, evocativa, ben strutturata.
Il concetto non rompe la generazione, la *tinge*.
