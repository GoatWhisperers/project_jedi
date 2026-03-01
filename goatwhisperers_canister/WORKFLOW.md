# GoatWhisperers Canister — ICP Project Workflow

Questo documento descrive il workflow completo per costruire e deployare
il canister ICP di Project Jedi: pubblicazioni on-chain, sistema di rewards
e raccolta fondi decentralizzata.

---

## Obiettivi

1. **Pubblicazioni on-chain** — ogni finding ha timestamp criptografico immutabile.
   Nessuno può appropriarsi della priorità di scoperta.
2. **Rewards** — incentivare contributi alla ricerca (dataset, validazioni, estensioni).
3. **Fundraising** — raccogliere ICP/cycles per finanziare i costi di compute.
4. **Frontend decentralizzato** — UI hostata su ICP, no server, no middleman.

---

## Architettura Canister

```
goatwhisperers_canister/
│
├── src/
│   ├── publications/          # Canister Motoko: pubblicazioni e timestamp
│   │   └── main.mo
│   ├── rewards/               # Canister Motoko: token e contributi
│   │   └── main.mo
│   ├── fundraising/           # Canister Motoko: raccolta fondi
│   │   └── main.mo
│   └── frontend/              # Asset canister: UI web
│       ├── index.html
│       ├── main.js
│       └── style.css
│
├── dfx.json                   # Config progetto ICP
├── package.json               # Dipendenze frontend (opzionale)
└── WORKFLOW.md                # Questo file
```

### Canister 1 — Publications

Registra ogni pubblicazione scientifica con:
- Hash SHA-256 del contenuto
- Timestamp on-chain (certificato dalla rete ICP)
- Autore (Principal ID)
- Metadati: titolo, abstract, concetti, modelli, link repo

**Garantisce:** priorità di scoperta verificabile e immutabile.
Chiunque può verificare che il finding esisteva in quel momento.

### Canister 2 — Rewards

Token fungibile semplice (non ICRC-1 nella prima versione, poi sì).
Emette reward per:
- Aggiunta di frasi fenomenologiche validate (10 token/frase accettata)
- Estensione della vector library a nuovi modelli (100 token/concept/modello)
- Esecuzione di blind evaluation sessions (50 token/sessione)
- Bug fix e contributi al codice (discrezionale, owner decide)

**Gate:** i token danno accesso anticipato a nuovi concetti e vettori,
e votare sulle priorità della roadmap.

### Canister 3 — Fundraising

Accetta donazioni in ICP.
- Pool trasparente, balance leggibile da chiunque
- Withdrawals firmati dall'owner (Principal ID)
- Log pubblico di ogni spesa (compute, hardware, tempo)
- Target campaigns: es. "Gemma3-27B extraction — target: 50 ICP"

### Canister 4 — Frontend (Asset Canister)

UI web hostata su ICP.
- Pagina pubblica: lista pubblicazioni + verifica hash
- Dashboard contributi: reward balance, storia contributi
- Fundraising page: campagne attive, progress bar, donazioni
- Accesso con Internet Identity (no password, no account)

---

## Step 1 — Setup Ambiente

### Installa dfx (DFINITY SDK)

```bash
# Installa dfx (versione stabile)
sh -ci "$(curl -fsSL https://internetcomputer.org/install.sh)"

# Verifica
dfx --version
# dfx 0.x.x

# Aggiungi al PATH se necessario
export PATH="$HOME/.local/share/dfx/bin:$PATH"
```

### Dipendenze opzionali

```bash
# Node.js (per frontend asset canister)
node --version   # >= 18 consigliato

# Motoko VSCode extension (per syntax highlighting)
# https://marketplace.visualstudio.com/items?itemName=dfinity-foundation.vscode-motoko
```

---

## Step 2 — Inizializza Progetto

```bash
cd /home/lele/codex-openai/project_jedi/goatwhisperers_canister

# Inizializza progetto ICP
dfx new goatwhisperers --no-frontend   # poi aggiungiamo frontend manuale
cd goatwhisperers

# Oppure, struttura manuale (più controllo):
mkdir -p src/publications src/rewards src/fundraising src/frontend
```

### dfx.json

```json
{
  "canisters": {
    "publications": {
      "main": "src/publications/main.mo",
      "type": "motoko"
    },
    "rewards": {
      "main": "src/rewards/main.mo",
      "type": "motoko"
    },
    "fundraising": {
      "main": "src/fundraising/main.mo",
      "type": "motoko"
    },
    "frontend": {
      "dependencies": ["publications", "rewards", "fundraising"],
      "source": ["src/frontend/"],
      "type": "assets",
      "args": ""
    }
  },
  "defaults": {
    "build": {
      "args": "",
      "packtool": ""
    }
  },
  "output_env_file": ".env",
  "version": 1
}
```

---

## Step 3 — Canister Publications (Motoko)

```motoko
// src/publications/main.mo

import Time "mo:base/Time";
import Array "mo:base/Array";
import Text "mo:base/Text";
import Principal "mo:base/Principal";

actor Publications {

  public type Publication = {
    id: Nat;
    title: Text;
    abstract_: Text;
    content_hash: Text;       // SHA-256 del documento completo
    concepts: [Text];         // es. ["hot_vs_cold", "luce_vs_buio"]
    models: [Text];           // es. ["Gemma3-1B-IT", "Gemma2-Uncensored"]
    author: Principal;
    timestamp: Int;           // nanoseconds since epoch (ICP time)
    repo_url: Text;
    doi: ?Text;               // opzionale, aggiunto dopo pubblicazione
  };

  stable var publications: [Publication] = [];
  stable var next_id: Nat = 0;

  // Pubblica un nuovo finding
  public shared(msg) func publish(
    title: Text,
    abstract_: Text,
    content_hash: Text,
    concepts: [Text],
    models: [Text],
    repo_url: Text
  ) : async Nat {
    let pub: Publication = {
      id = next_id;
      title = title;
      abstract_ = abstract_;
      content_hash = content_hash;
      concepts = concepts;
      models = models;
      author = msg.caller;
      timestamp = Time.now();
      repo_url = repo_url;
      doi = null;
    };
    publications := Array.append(publications, [pub]);
    next_id += 1;
    return next_id - 1;
  };

  // Lista tutte le pubblicazioni
  public query func list() : async [Publication] {
    return publications;
  };

  // Verifica hash (chiunque può verificare)
  public query func verify(id: Nat, hash: Text) : async Bool {
    if (id >= publications.size()) return false;
    return publications[id].content_hash == hash;
  };

  // Aggiunge DOI post-pubblicazione (solo author)
  public shared(msg) func add_doi(id: Nat, doi: Text) : async Bool {
    if (id >= publications.size()) return false;
    if (publications[id].author != msg.caller) return false;
    // update DOI — in Motoko array stable richiede ricostruzione
    // (implementazione completa nella versione finale)
    return true;
  };
}
```

---

## Step 4 — Canister Rewards (Motoko)

```motoko
// src/rewards/main.mo — schema base, da espandere

import Principal "mo:base/Principal";
import HashMap "mo:base/HashMap";
import Nat "mo:base/Nat";
import Hash "mo:base/Hash";

actor Rewards {

  // Balance per ogni contributor
  stable var balances_entries: [(Principal, Nat)] = [];
  var balances = HashMap.fromIter<Principal, Nat>(
    balances_entries.vals(), 16, Principal.equal, Principal.hash
  );

  let OWNER: Principal = Principal.fromText("SOSTITUIRE_CON_PRINCIPAL_ID");

  // Mint reward (solo owner)
  public shared(msg) func mint(to: Principal, amount: Nat) : async Bool {
    if (msg.caller != OWNER) return false;
    let current = switch (balances.get(to)) { case null 0; case (?v) v };
    balances.put(to, current + amount);
    return true;
  };

  // Leggi balance
  public query func balance_of(who: Principal) : async Nat {
    switch (balances.get(who)) { case null 0; case (?v) v }
  };

  // Lista top contributors
  public query func leaderboard() : async [(Principal, Nat)] {
    // implementazione nella versione finale
    return [];
  };

  // Upgrade hook
  system func preupgrade() {
    balances_entries := balances.entries() |> (func(it) {
      var arr: [(Principal, Nat)] = [];
      for (entry in it) { arr := [(entry.0, entry.1)] };
      arr
    })(balances.entries());
  };
}
```

---

## Step 5 — Canister Fundraising (Motoko)

```motoko
// src/fundraising/main.mo — schema base

import Time "mo:base/Time";
import Array "mo:base/Array";
import Principal "mo:base/Principal";

actor Fundraising {

  public type Campaign = {
    id: Nat;
    title: Text;
    description: Text;
    target_icp: Nat;      // in e8s (1 ICP = 100_000_000 e8s)
    raised_icp: Nat;
    active: Bool;
    created_at: Int;
  };

  public type Donation = {
    campaign_id: Nat;
    donor: Principal;
    amount: Nat;
    timestamp: Int;
    message: Text;
  };

  stable var campaigns: [Campaign] = [];
  stable var donations: [Donation] = [];
  stable var next_id: Nat = 0;

  let OWNER: Principal = Principal.fromText("SOSTITUIRE_CON_PRINCIPAL_ID");

  // Crea campagna (solo owner)
  public shared(msg) func create_campaign(
    title: Text,
    description: Text,
    target_icp: Nat
  ) : async Nat {
    if (msg.caller != OWNER) return 0;
    let c: Campaign = {
      id = next_id;
      title = title;
      description = description;
      target_icp = target_icp;
      raised_icp = 0;
      active = true;
      created_at = Time.now();
    };
    campaigns := Array.append(campaigns, [c]);
    next_id += 1;
    return next_id - 1;
  };

  // Lista campagne attive
  public query func list_campaigns() : async [Campaign] {
    return campaigns;
  };

  // Lista donazioni (trasparenza totale)
  public query func list_donations() : async [Donation] {
    return donations;
  };
}
```

---

## Step 6 — Deploy Locale (Testing)

```bash
cd goatwhisperers_canister

# Avvia replica locale ICP
dfx start --background --clean

# Deploy tutti i canister in locale
dfx deploy

# Output: canister IDs locali + URL frontend
# es. http://127.0.0.1:4943/?canisterId=<id>

# Test publications canister
dfx canister call publications publish '(
  "Stability vs Correctness in LLM Concept Vectors",
  "We show that early-layer vectors achieve near-perfect bootstrap stability...",
  "sha256:abc123...",
  vec { "hot_vs_cold"; "luce_vs_buio" },
  vec { "Gemma3-1B-IT"; "Gemma2-Uncensored" },
  "https://github.com/GoatWhisperers/project_jedi"
)'

# Leggi pubblicazioni
dfx canister call publications list '()'

# Stop replica locale
dfx stop
```

---

## Step 7 — Deploy su ICP Mainnet

```bash
# Assicurati di avere cycles (ICP → cycles tramite NNS o exchange)
# ~1-2 ICP per deploy iniziale

# Recupera il tuo Principal ID
dfx identity get-principal

# Controlla balance cycles
dfx wallet balance

# Deploy su mainnet
dfx deploy --network ic

# Output: canister IDs mainnet
# es. https://<canister-id>.icp0.io  (frontend)
#     <canister-id>.ic0.app          (API)

# Salva i canister ID — servono per interagire dopo
dfx canister --network ic id publications
dfx canister --network ic id rewards
dfx canister --network ic id fundraising
dfx canister --network ic id frontend
```

---

## Step 8 — Prima Pubblicazione

Una volta deployato, la prima pubblicazione dovrebbe essere il paper/post
che descrive Project Jedi:

```bash
# Calcola hash del documento
sha256sum experiments/architecture.md

# Pubblica on-chain
dfx canister --network ic call publications publish '(
  "Project Jedi: Phenomenological Concept Vectors for Transformer Steering",
  "We extract linear concept directions from transformer hidden states...",
  "sha256:<hash>",
  vec { "hot_vs_cold"; "luce_vs_buio"; "calma_vs_allerta"; ... },
  vec { "Gemma3-1B-IT"; "Gemma2-Uncensored" },
  "https://github.com/GoatWhisperers/project_jedi"
)'
```

Timestamp immutabile. Chiunque, in qualsiasi momento futuro, può
verificare che questa ricerca esisteva a questa data.

---

## Roadmap Canister

| Fase | Cosa | Priorità |
|------|------|----------|
| v0.1 | Publications canister + deploy mainnet | Alta |
| v0.2 | Frontend base: lista pub + verifica hash | Alta |
| v0.3 | Rewards canister + mint manuale | Media |
| v0.4 | Fundraising canister + prima campagna | Media |
| v0.5 | ICRC-1 token standard per rewards | Bassa |
| v1.0 | DAO voting sulla roadmap | Futura |

---

## Note Tecniche

- **Motoko vs Rust:** Motoko è più semplice per iniziare, Rust dà più controllo.
  Per questi canister Motoko è sufficiente.
- **Stable variables:** Essenziali — i dati sopravvivono agli upgrade del canister.
- **Cycles:** Ogni operazione costa cycles. Stima: ~0.5 ICP/anno per canister piccolo.
- **Internet Identity:** Sistema di autenticazione nativo ICP, senza password.
  Da integrare nel frontend per identificare i contributor.
- **ICRC-1:** Standard token ICP (equivalente ERC-20 su Ethereum).
  Adottarlo nella v0.5 permette listing su exchange decentralizzati.

---

## Risorse

- Documentazione ICP: https://internetcomputer.org/docs
- Motoko Language: https://internetcomputer.org/docs/current/motoko/main/motoko
- dfx CLI reference: https://internetcomputer.org/docs/current/developer-docs/developer-tools/cli-tools/cli-reference
- Motoko Playground (test online): https://play.motoko.org
- ICP Dashboard (canister info): https://dashboard.internetcomputer.org
