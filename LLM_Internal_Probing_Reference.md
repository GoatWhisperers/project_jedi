# LLM Internal Probing — Technical Reference for Code Generation

## Document Purpose and Scope

This document is a complete technical reference for generating Python code that extracts, analyzes, manipulates, and maps the internal representations of Large Language Models during inference. It is intended as a knowledge base for AI coding assistants (Claude Code, Codex, Copilot, or any LLM-based code generator) so they can produce correct, efficient, and theoretically grounded code for any probing task on transformer-based language models.

The target hardware is a single-GPU setup with 16–48 GB VRAM running models in the 7B–14B parameter range. All code patterns assume PyTorch and HuggingFace Transformers. ROCm (AMD) and CUDA (NVIDIA) are both supported — PyTorch abstracts the difference through the `torch.cuda` API for both backends.

---

## Part 1 — What Lives Inside a Transformer During Inference

### 1.1 The Forward Pass as a Data Pipeline

When a transformer-based LLM processes an input, the following happens in strict sequence:

```
Input text
    │
    ▼
TOKENIZER — converts text to token IDs (integers)
    │         shape: (batch_size, seq_len)
    │         dtype: int64
    │         lives on: CPU initially, then moved to GPU
    ▼
EMBEDDING LAYER — converts token IDs to dense vectors
    │         shape: (batch_size, seq_len, hidden_dim)
    │         dtype: float16 or bfloat16
    │         This is hidden_states[0] — the "pre-transformer" representation
    │         Each token is now a vector of `hidden_dim` floating point numbers
    │         hidden_dim is typically 3072 (7B), 3584 (9B), 4096 (12B–13B), 5120 (14B+)
    ▼
TRANSFORMER LAYER 0
    │   ┌─────────────────────────────────────────┐
    │   │ 1. Layer Norm (RMSNorm in modern models) │
    │   │ 2. Self-Attention                        │
    │   │    - Q, K, V projections                 │
    │   │    - Attention weight computation         │
    │   │    - Weighted value aggregation           │
    │   │    - Output projection                    │
    │   │ 3. Residual connection (add input back)   │
    │   │ 4. Layer Norm                             │
    │   │ 5. MLP (Feed-Forward Network)             │
    │   │    - Up projection (hidden_dim → ffn_dim) │
    │   │    - Activation function (SiLU/GELU)      │
    │   │    - Gate projection (in gated models)     │
    │   │    - Down projection (ffn_dim → hidden_dim)│
    │   │ 6. Residual connection (add input back)   │
    │   └─────────────────────────────────────────┘
    │         output shape: (batch_size, seq_len, hidden_dim)
    │         This is hidden_states[1]
    ▼
TRANSFORMER LAYER 1 → hidden_states[2]
    ▼
    ... (repeat for N layers)
    ▼
TRANSFORMER LAYER N-1 → hidden_states[N]
    ▼
FINAL LAYER NORM
    ▼
LM HEAD — projects hidden_dim → vocab_size to produce logits
    │         shape: (batch_size, seq_len, vocab_size)
    │         vocab_size is typically 32000–256000
    ▼
SOFTMAX → probability distribution over next token
```

**Critical facts for code generation:**

- `hidden_states` is a tuple of `(N + 1)` tensors, where N = number of transformer layers
- `hidden_states[0]` is the embedding output (before any transformer layer)
- `hidden_states[i]` for i in 1..N is the output AFTER transformer layer (i-1)
- `hidden_states[-1]` is the output of the last transformer layer (before LM head)
- Every tensor in `hidden_states` has identical shape: `(batch_size, seq_len, hidden_dim)`
- The residual connections mean each layer's output ADDS to the previous representation; information accumulates, it is not replaced

### 1.2 What a Single Hidden State Vector Contains

A single vector extracted from position `(batch=0, token=t, layer=L)` is a 1D tensor of `hidden_dim` floating-point numbers. For example, for a model with hidden_dim=4096:

```python
vector = hidden_states[L][0, t, :]  # shape: (4096,)
# This is a point in a 4096-dimensional vector space
```

This vector simultaneously encodes:

1. **Token identity** — what word/subword is at this position
2. **Positional information** — where in the sequence this token sits
3. **Contextual meaning** — how surrounding tokens modify its meaning
4. **Syntactic role** — grammatical function in the sentence
5. **Semantic content** — conceptual meaning
6. **Pragmatic context** — discourse-level information
7. **Task-relevant features** — information needed to predict the next token
8. **World knowledge** — facts and relationships activated by the context

All of this information is encoded **simultaneously** in the same vector through **superposition**: different concepts correspond to different directions in the high-dimensional space, and because the space has thousands of dimensions, many nearly-orthogonal directions can coexist without interfering.

### 1.3 The Geometry of Meaning

Meaning in transformer hidden states is **directional, not dimensional**.

- A **direction** is a unit vector in the hidden_dim-dimensional space: a vector of hidden_dim numbers whose L2 norm equals 1.
- A **concept** corresponds to a direction (or a low-dimensional subspace) in this space.
- The **strength** of a concept's presence in a hidden state is measured by projecting the hidden state onto the concept's direction (dot product).
- Two concepts are **independent** if their directions are orthogonal (dot product ≈ 0).
- Two concepts are **related** if their directions have significant cosine similarity.

This means:

```python
# If concept_direction is a unit vector representing "temperature"
# and hidden_state is the activation at some layer:
temperature_strength = torch.dot(hidden_state, concept_direction)
# temperature_strength is a SCALAR:
#   positive = "hot" direction
#   negative = "cold" direction
#   near zero = concept not present
#   magnitude = how strongly the concept is activated
```

### 1.4 The Depth-Semantics Correspondence

Different layers of the transformer encode qualitatively different types of information. This is not a hypothesis — it is an empirically verified property of all transformer LLMs studied to date. The general pattern is:

| Layer Depth | What is Encoded | Examples |
|---|---|---|
| Layers 0–15% | Token identity, positional encoding, basic syntactic features | Part of speech, word morphology, local word order |
| Layers 15–40% | Syntactic structure, grammatical relations | Subject-verb agreement, clause boundaries, dependency relations |
| Layers 40–70% | Semantic meaning, conceptual categories, entity types | Animal vs object, emotion type, spatial relations, abstract concepts |
| Layers 70–90% | World knowledge, factual associations, relational reasoning | Capital cities, physical properties, causal relationships |
| Layers 90–100% | Task-specific features, next-token prediction | Output formatting, instruction following, style adaptation |

Percentages are approximate and vary by model family. The key point: **information type determines optimal probing depth**.

For a model with 40 layers:
- Surface features peak around layers 2–6
- Syntactic features peak around layers 6–16
- Semantic features peak around layers 16–28
- Deep knowledge peaks around layers 28–36
- Task features peak around layers 36–40

### 1.5 Special Token Positions

Not all token positions carry the same information:

- **Last token position** (`seq_len - 1`): In causal (autoregressive) models, this position has attended to ALL previous tokens and contains the richest contextual representation. This is the standard extraction point for sentence-level probing.
- **First token position** (0): Often contains global summary information due to attention sink effects.
- **Subject/object token positions**: Contain entity-specific information, useful for probing factual knowledge.
- **Verb positions**: Contain relational and action information.

For most probing tasks, **extract from the last token position**. For entity-specific probing (e.g., "what does the model know about X?"), extract from the token position corresponding to X.

---

## Part 2 — Model-Specific Architecture Details

### 2.1 How to Determine Model Architecture Programmatically

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_name)

# Universal attributes:
num_layers = config.num_hidden_layers          # N transformer layers
hidden_dim = config.hidden_size                 # Dimension of hidden states
vocab_size = config.vocab_size                  # Size of token vocabulary

# Common but model-family-specific:
num_heads = config.num_attention_heads          # Number of attention heads
num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)  # GQA heads
intermediate_size = config.intermediate_size    # FFN intermediate dimension
head_dim = hidden_dim // num_heads              # Dimension per attention head

print(f"Model: {model_name}")
print(f"Layers: {num_layers}, Hidden dim: {hidden_dim}")
print(f"Attention heads: {num_heads}, KV heads: {num_kv_heads}")
print(f"FFN intermediate size: {intermediate_size}")
print(f"Parameters (approx): {num_layers * (4 * hidden_dim**2 + 2 * hidden_dim * intermediate_size) / 1e9:.1f}B")
```

### 2.2 Model Family Reference Table

| Model Family | Layer Access Path | Attention Module | MLP Module | Layer Norm | Notes |
|---|---|---|---|---|---|
| Llama 2/3, Code Llama | `model.model.layers[i]` | `.self_attn` | `.mlp` | RMSNorm | GQA in 3.x |
| Gemma 1/2/3 | `model.model.layers[i]` | `.self_attn` | `.mlp` | RMSNorm | Same as Llama structure |
| Mistral, Mixtral | `model.model.layers[i]` | `.self_attn` | `.mlp` | RMSNorm | Mixtral: MoE in MLP |
| Qwen 2/2.5 | `model.model.layers[i]` | `.self_attn` | `.mlp` | RMSNorm | Same as Llama structure |
| Phi-3/4 | `model.model.layers[i]` | `.self_attn` | `.mlp` | RMSNorm (Phi-3), LayerNorm (Phi-4) | |
| GPT-NeoX (Pythia) | `model.gpt_neox.layers[i]` | `.attention` | `.mlp` | LayerNorm | Parallel attention+MLP |
| Falcon | `model.transformer.h[i]` | `.self_attention` | `.mlp` | LayerNorm | |

**Generic layer access pattern:**

```python
def get_transformer_layers(model):
    """Return the list of transformer layer modules, model-agnostic."""
    # Try common paths in order of likelihood
    for attr_path in [
        "model.layers",           # Llama, Gemma, Mistral, Qwen
        "model.model.layers",     # When wrapped in ForCausalLM
        "transformer.h",          # GPT-2, Falcon
        "gpt_neox.layers",        # GPT-NeoX, Pythia
        "transformer.blocks",     # MPT
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer layers for {type(model).__name__}")
```

### 2.3 VRAM Budget Planning

```
Model parameters (in billions) × bytes per parameter = VRAM for weights

float32:  4 bytes/param  → 12B model = ~48 GB  (won't fit MI50)
float16:  2 bytes/param  → 12B model = ~24 GB  (fits MI50 with ~8 GB margin)
bfloat16: 2 bytes/param  → 12B model = ~24 GB  (same, better for training)
int8:     1 byte/param   → 12B model = ~12 GB  (fits easily, slight quality loss)
int4:     0.5 byte/param → 12B model = ~6 GB   (fits any GPU, notable quality loss)

Additional VRAM during inference:
- KV cache: ~2 × num_layers × 2 × num_kv_heads × head_dim × seq_len × 2 bytes
  For 12B model, seq_len=512: ~0.5 GB
  For 12B model, seq_len=4096: ~4 GB
- Activation tensors (with output_hidden_states=True):
  (num_layers + 1) × batch_size × seq_len × hidden_dim × 2 bytes
  For 12B (40 layers), batch=1, seq_len=512: ~0.3 GB
  WARNING: this grows linearly with batch size and sequence length
```

**Decision rules for code generation:**

1. If model_params × 2 (for FP16) < 0.75 × VRAM → use `torch_dtype=torch.float16`
2. If model_params × 2 > 0.75 × VRAM → use `load_in_8bit=True` (requires bitsandbytes)
3. If model_params × 1 > 0.75 × VRAM → use `load_in_4bit=True` (requires bitsandbytes)
4. Always leave 20–25% VRAM margin for KV cache, activations, and probing operations
5. For probing tasks, immediately move extracted tensors to CPU (`.cpu()`) and delete GPU tensors

---

## Part 3 — Extraction Patterns

### 3.1 Pattern: Extract All Hidden States for a Single Input

This is the fundamental extraction pattern used by all probing methods.

```python
def extract_all_hidden_states(model, tokenizer, text, device=None):
    """
    Extract hidden states from all layers for a single text input.
    
    Returns:
        hidden_states: numpy array of shape (num_layers+1, seq_len, hidden_dim)
        tokens: list of token strings for reference
    """
    if device is None:
        device = next(model.parameters()).device
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Stack all hidden states and move to CPU immediately
    # Shape: (num_layers+1, seq_len, hidden_dim)
    hidden_states = torch.stack(outputs.hidden_states).squeeze(1).cpu().numpy()
    
    # Decode tokens for reference
    tokens = [tokenizer.decode(tid) for tid in inputs["input_ids"][0]]
    
    # Free GPU memory
    del outputs
    torch.cuda.empty_cache()
    
    return hidden_states, tokens
```

### 3.2 Pattern: Extract Hidden States at a Specific Layer for Multiple Texts

This is the workhorse pattern for probing experiments — efficient batch extraction at a single layer.

```python
def extract_at_layer(model, tokenizer, texts, layer_idx, 
                     token_position="last", batch_size=4, device=None):
    """
    Extract hidden state vectors at a specific layer for multiple texts.
    
    Args:
        model: HuggingFace model with output_hidden_states=True
        tokenizer: corresponding tokenizer
        texts: list of strings
        layer_idx: which layer to extract from (0 = embedding, 1..N = transformer layers)
        token_position: "last" (default), "first", "mean", or int index
        batch_size: process this many texts at once (reduce if OOM)
    
    Returns:
        vectors: numpy array of shape (len(texts), hidden_dim)
    """
    if device is None:
        device = next(model.parameters()).device
    
    all_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
        
        if token_position == "last":
            # Find the actual last token (before padding) for each item
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            vectors = torch.stack([
                hidden[b, seq_lengths[b], :] for b in range(hidden.size(0))
            ])
        elif token_position == "first":
            vectors = hidden[:, 0, :]
        elif token_position == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            vectors = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        elif isinstance(token_position, int):
            vectors = hidden[:, token_position, :]
        else:
            raise ValueError(f"Unknown token_position: {token_position}")
        
        all_vectors.append(vectors.cpu().numpy())
        
        del outputs, hidden, vectors
        torch.cuda.empty_cache()
    
    return np.vstack(all_vectors)
```

### 3.3 Pattern: Hook-Based Extraction (For Sub-Layer Components)

When you need to extract not just the layer output but the internal components (attention output, MLP output, attention weights), use hooks.

```python
class ComponentExtractor:
    """
    Extract sub-layer components using forward hooks.
    
    Usage:
        extractor = ComponentExtractor(model)
        extractor.attach_to_layers([10, 20, 30], components=["attn", "mlp"])
        
        outputs = model(**inputs)
        
        attn_out_layer20 = extractor.get("layer_20_attn")  # numpy array
        mlp_out_layer20 = extractor.get("layer_20_mlp")    # numpy array
        
        extractor.detach()
    """
    
    def __init__(self, model):
        self.model = model
        self.layers = get_transformer_layers(model)
        self._activations = {}
        self._hooks = []
    
    def attach_to_layers(self, layer_indices, components=("attn", "mlp")):
        """
        Attach hooks to specified layers and components.
        
        components can include:
            "attn"  — output of the self-attention module
            "mlp"   — output of the MLP/FFN module
            "input" — input to the transformer layer (= previous layer's output)
        """
        for idx in layer_indices:
            layer = self.layers[idx]
            
            if "attn" in components:
                # The self_attn module - try common attribute names
                attn_module = None
                for attr in ["self_attn", "attention", "self_attention"]:
                    if hasattr(layer, attr):
                        attn_module = getattr(layer, attr)
                        break
                if attn_module:
                    hook = attn_module.register_forward_hook(
                        self._make_hook(f"layer_{idx}_attn")
                    )
                    self._hooks.append(hook)
            
            if "mlp" in components:
                mlp_module = None
                for attr in ["mlp", "feed_forward", "ffn"]:
                    if hasattr(layer, attr):
                        mlp_module = getattr(layer, attr)
                        break
                if mlp_module:
                    hook = mlp_module.register_forward_hook(
                        self._make_hook(f"layer_{idx}_mlp")
                    )
                    self._hooks.append(hook)
            
            if "input" in components:
                hook = layer.register_forward_pre_hook(
                    self._make_pre_hook(f"layer_{idx}_input")
                )
                self._hooks.append(hook)
    
    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._activations[name] = output[0].detach().cpu()
            else:
                self._activations[name] = output.detach().cpu()
        return hook_fn
    
    def _make_pre_hook(self, name):
        def hook_fn(module, input):
            if isinstance(input, tuple):
                self._activations[name] = input[0].detach().cpu()
            else:
                self._activations[name] = input.detach().cpu()
        return hook_fn
    
    def get(self, name):
        """Get extracted activation as numpy array."""
        return self._activations.get(name, None)
    
    def get_all(self):
        """Get all extracted activations."""
        return dict(self._activations)
    
    def clear(self):
        self._activations.clear()
    
    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.clear()
```

### 3.4 Pattern: Attention Weight Extraction

Attention weights reveal which tokens the model considers related. Requires `output_attentions=True`.

```python
def extract_attention_weights(model, tokenizer, text, device=None):
    """
    Extract attention weight matrices from all layers.
    
    Returns:
        attentions: numpy array of shape (num_layers, num_heads, seq_len, seq_len)
        tokens: list of token strings
    
    NOTE: Each attention matrix [h, i, j] = how much token i attends to token j
          Rows sum to 1.0 (softmax over keys)
    """
    if device is None:
        device = next(model.parameters()).device
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is a tuple of (num_layers) tensors
    # Each tensor shape: (batch, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1).cpu().numpy()
    tokens = [tokenizer.decode(tid) for tid in inputs["input_ids"][0]]
    
    del outputs
    torch.cuda.empty_cache()
    
    return attentions, tokens
```

**WARNING on attention weights:** Extracting attention weights increases VRAM usage significantly because the model must store all N × H attention matrices. For long sequences on memory-constrained GPUs, extract attention for only specific layers using hooks instead.

---

## Part 4 — Analysis Patterns

### 4.1 Pattern: Contrastive Concept Vector Extraction

The most fundamental analysis: find the direction in activation space that corresponds to a specific concept.

**Theoretical basis:** If you have two sets of texts that differ primarily in one concept (e.g., "hot" vs "cold"), the difference of their mean activation vectors isolates the direction corresponding to that concept. This works because:
- All other shared features (grammar, context, language) cancel out in the subtraction
- The concept-specific feature adds constructively in one group and subtracts in the other
- The resulting difference vector points in the direction of maximum concept discrimination

```python
def extract_concept_vector(model, tokenizer, positive_texts, negative_texts, 
                           layer_idx, token_position="last", normalize=True):
    """
    Extract a concept vector by contrastive analysis.
    
    Args:
        positive_texts: texts where the concept is present/strong (e.g., hot things)
        negative_texts: texts where the concept is absent/opposite (e.g., cold things)
        layer_idx: which layer to extract from
        normalize: if True, return unit vector (direction only)
    
    Returns:
        concept_vector: numpy array of shape (hidden_dim,)
        metadata: dict with extraction details
    
    IMPORTANT DATA QUALITY RULES:
    1. Minimum 8 texts per class, ideally 20+
    2. Texts should vary in topic, structure, and vocabulary
    3. The ONLY consistent difference should be the target concept
    4. Avoid having the concept appear as a specific token — vary the expression
       BAD:  ["The hot water", "The hot sand", "The hot metal"]  (all contain "hot")
       GOOD: ["The water is boiling", "The desert scorches", "Molten metal flows"]
    5. Match text length roughly between classes
    """
    pos_vecs = extract_at_layer(model, tokenizer, positive_texts, layer_idx, token_position)
    neg_vecs = extract_at_layer(model, tokenizer, negative_texts, layer_idx, token_position)
    
    pos_mean = pos_vecs.mean(axis=0)
    neg_mean = neg_vecs.mean(axis=0)
    
    concept_vector = pos_mean - neg_mean
    magnitude = np.linalg.norm(concept_vector)
    
    if normalize and magnitude > 0:
        concept_vector = concept_vector / magnitude
    
    metadata = {
        "layer_idx": layer_idx,
        "num_positive": len(positive_texts),
        "num_negative": len(negative_texts),
        "magnitude_before_norm": float(magnitude),
        "pos_mean_norm": float(np.linalg.norm(pos_mean)),
        "neg_mean_norm": float(np.linalg.norm(neg_mean)),
        "cosine_pos_neg_means": float(
            np.dot(pos_mean, neg_mean) / (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean))
        ),
    }
    
    return concept_vector, metadata
```

### 4.2 Pattern: Linear Probe (Supervised Concept Detection)

A linear probe is a logistic regression classifier trained on hidden states to predict whether a concept is present. It serves two purposes:
1. **Measuring** whether a layer encodes a concept (high accuracy = yes)
2. **Extracting** the concept direction (the classifier's weight vector IS the concept direction)

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_linear_probe(X, y, probe_type="classification", cv_folds=5, C=1.0):
    """
    Train a linear probe on activation vectors.
    
    Args:
        X: numpy array of shape (n_samples, hidden_dim) — the activation vectors
        y: numpy array of shape (n_samples,) — labels
           For classification: binary (0/1) or multiclass (int)
           For regression: continuous float values
        probe_type: "classification" or "regression"
        cv_folds: number of cross-validation folds
        C: regularization strength (lower = more regularization)
    
    Returns:
        probe: trained sklearn model
        scores: cross-validation scores
        concept_vector: the weight vector (= concept direction for binary classification)
    
    IMPORTANT:
    - Regularization (C parameter) matters: too high → overfitting to spurious features,
      too low → underfitting. Default C=1.0 is usually good. If accuracy is suspiciously 
      high (>0.98 with small datasets), reduce C.
    - Minimum 6 samples per class for meaningful cross-validation.
    - StandardScaler is optional for probing (hidden states are already roughly normalized
      by RMSNorm), but can help with numerical stability for very deep layers.
    """
    if probe_type == "classification":
        probe = LogisticRegression(max_iter=2000, C=C, solver="lbfgs")
        scoring = "accuracy"
    elif probe_type == "regression":
        probe = Ridge(alpha=1.0/C)
        scoring = "r2"
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
    
    # Cross-validation
    if probe_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = cross_val_score(probe, X, y, cv=cv, scoring=scoring)
    
    # Train on full data to get the concept vector
    probe.fit(X, y)
    
    # Extract concept direction
    if probe_type == "classification":
        if len(probe.classes_) == 2:
            concept_vector = probe.coef_[0]  # shape: (hidden_dim,)
        else:
            concept_vector = probe.coef_      # shape: (n_classes, hidden_dim)
    else:
        concept_vector = probe.coef_          # shape: (hidden_dim,)
    
    return probe, scores, concept_vector


def probe_across_all_layers(model, tokenizer, positive_texts, negative_texts,
                            token_position="last"):
    """
    Run linear probes across all layers to find WHERE a concept is encoded.
    
    Returns:
        results: dict mapping layer_idx → {accuracy, std, concept_vector}
    """
    num_layers = model.config.num_hidden_layers + 1
    results = {}
    
    for layer_idx in range(num_layers):
        pos_vecs = extract_at_layer(model, tokenizer, positive_texts, layer_idx, token_position)
        neg_vecs = extract_at_layer(model, tokenizer, negative_texts, layer_idx, token_position)
        
        X = np.vstack([pos_vecs, neg_vecs])
        y = np.array([1] * len(pos_vecs) + [0] * len(neg_vecs))
        
        n_folds = min(5, min(len(pos_vecs), len(neg_vecs)))
        if n_folds < 2:
            results[layer_idx] = {"accuracy": 0.5, "std": 0.0, "concept_vector": None}
            continue
        
        probe, scores, concept_vector = train_linear_probe(X, y, cv_folds=n_folds)
        
        results[layer_idx] = {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "concept_vector": concept_vector / np.linalg.norm(concept_vector),
        }
    
    return results
```

### 4.3 Pattern: Concept Projection (Measuring Concept Strength)

Once you have a concept vector, project new inputs onto it to measure concept presence.

```python
def project_onto_concept(model, tokenizer, texts, concept_vector, layer_idx,
                         token_position="last"):
    """
    Project texts onto a concept direction and return scalar strengths.
    
    Args:
        texts: list of strings to analyze
        concept_vector: unit vector defining the concept direction
        layer_idx: layer to extract from (should match the layer the concept was extracted from)
    
    Returns:
        projections: numpy array of shape (len(texts),) — signed scalar values
            positive = aligned with concept direction (e.g., "hot")
            negative = opposite to concept direction (e.g., "cold")
            near zero = concept not present
    """
    # Ensure concept_vector is a unit vector
    concept_unit = concept_vector / np.linalg.norm(concept_vector)
    
    vectors = extract_at_layer(model, tokenizer, texts, layer_idx, token_position)
    
    # Dot product of each vector with the concept direction
    projections = vectors @ concept_unit
    
    return projections


def build_concept_profile(model, tokenizer, text, concept_library, layer_idx,
                          token_position="last"):
    """
    Build a multi-concept profile for a single text.
    
    Args:
        concept_library: dict mapping concept_name → concept_vector (unit vectors)
    
    Returns:
        profile: dict mapping concept_name → float (projection strength)
    """
    vector = extract_at_layer(model, tokenizer, [text], layer_idx, token_position)[0]
    
    profile = {}
    for concept_name, concept_vec in concept_library.items():
        concept_unit = concept_vec / np.linalg.norm(concept_vec)
        profile[concept_name] = float(np.dot(vector, concept_unit))
    
    return profile
```

### 4.4 Pattern: Cosine Similarity Analysis

```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(vectors, labels=None):
    """
    Compute pairwise cosine similarity between vectors.
    
    Args:
        vectors: numpy array of shape (N, hidden_dim)
        labels: optional list of N labels for display
    
    Returns:
        sim_matrix: numpy array of shape (N, N) with values in [-1, 1]
    """
    sim_matrix = cosine_similarity(vectors)
    return sim_matrix


def compare_representations_across_layers(model, tokenizer, text, layer_indices):
    """
    Show how the representation of a text evolves across layers.
    
    Returns: similarity matrix between the same text's representation at different layers.
    Diagonal is always 1.0. Off-diagonal shows how much the representation changes.
    """
    all_hidden, tokens = extract_all_hidden_states(model, tokenizer, text)
    
    # Extract last-token vector at each specified layer
    vectors = np.array([all_hidden[l, -1, :] for l in layer_indices])
    
    return cosine_similarity(vectors)
```

### 4.5 Pattern: Dimensionality Reduction and Visualization

```python
import umap
from sklearn.decomposition import PCA

def visualize_concept_space(vectors_dict, method="umap", n_components=2, **kwargs):
    """
    Visualize groups of vectors in 2D or 3D space.
    
    Args:
        vectors_dict: dict mapping label → numpy array of shape (N_i, hidden_dim)
                      e.g., {"hot": hot_vectors, "cold": cold_vectors, ...}
        method: "umap" or "pca"
    
    Returns:
        embedding: numpy array of shape (total_N, n_components)
        labels: list of labels for each point
        colors: list of color indices for each point
    """
    all_vectors = []
    labels = []
    color_indices = []
    
    for i, (label, vecs) in enumerate(vectors_dict.items()):
        all_vectors.append(vecs)
        labels.extend([label] * len(vecs))
        color_indices.extend([i] * len(vecs))
    
    all_vectors = np.vstack(all_vectors)
    
    if method == "umap":
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get("n_neighbors", min(15, len(all_vectors) - 1)),
            min_dist=kwargs.get("min_dist", 0.1),
            metric=kwargs.get("metric", "cosine"),
            random_state=42
        )
        embedding = reducer.fit_transform(all_vectors)
    elif method == "pca":
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(all_vectors)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return embedding, labels, color_indices
```

---

## Part 5 — Intervention Patterns

### 5.1 Pattern: Activation Steering (Concept Injection)

This is the pattern that directly corresponds to the Depth-Appropriate Injection
described in NeuroOS theory. You modify the model's internal representations at a
specific layer to steer its behavior in a chosen conceptual direction.

```python
class ActivationSteerer:
    """
    Injects a steering vector at a specific layer during generation.
    
    This modifies the model's hidden states at the target layer by adding
    a scaled concept vector, effectively "pushing" the model's representation
    in a chosen direction.
    
    The steering equation at the target layer is:
        h_modified = h_original + alpha * steering_vector
    
    where:
        h_original: the natural hidden state (batch, seq_len, hidden_dim)
        steering_vector: the concept direction (hidden_dim,)
        alpha: scalar controlling injection strength
    
    Alpha guidelines:
        0.5–1.0:  subtle nudge, often undetectable in output
        1.0–3.0:  noticeable influence on word choice and topic
        3.0–6.0:  strong steering, clearly redirects content
        6.0–10.0: dominant influence, may cause incoherence
        >10.0:    likely to break generation quality
    
    The optimal alpha depends on the norm of the layer's hidden states.
    A more principled approach is to scale alpha relative to the average
    hidden state norm at that layer:
        alpha_normalized = alpha_raw * mean_hidden_norm / steering_vector_norm
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_transformer_layers(model)
        self._hooks = []
    
    def steer_and_generate(self, prompt, steering_vector, layer_idx, alpha=3.0,
                           max_new_tokens=100, temperature=0.7, do_sample=True,
                           apply_to="all"):
        """
        Generate text with a steering vector injected at a specific layer.
        
        Args:
            steering_vector: numpy array of shape (hidden_dim,)
            layer_idx: which transformer layer to inject at
            alpha: steering strength
            apply_to: "all" (every token), "new" (only generated tokens), 
                      "last" (only last token position)
        
        Returns:
            generated_text: the generated continuation (excluding prompt)
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        steer_tensor = torch.tensor(steering_vector, dtype=dtype, device=device)
        prompt_len = len(self.tokenizer.encode(prompt))
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            modified = hidden.clone()
            
            if apply_to == "all":
                modified += alpha * steer_tensor
            elif apply_to == "new":
                # Only steer positions beyond the original prompt
                if modified.shape[1] > prompt_len:
                    modified[:, prompt_len:, :] += alpha * steer_tensor
            elif apply_to == "last":
                modified[:, -1, :] += alpha * steer_tensor
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        
        # Attach hook
        hook = self.layers[layer_idx].register_forward_hook(hook_fn)
        self._hooks.append(hook)
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        
        # Clean up
        self._remove_hooks()
        
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text
    
    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def run_steering_experiment(model, tokenizer, prompt, concept_vector, layer_idx,
                            alphas=None):
    """
    Run a systematic steering experiment varying alpha strength.
    
    Returns:
        results: list of (alpha, generated_text) tuples
    """
    if alphas is None:
        alphas = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
    
    steerer = ActivationSteerer(model, tokenizer)
    results = []
    
    for alpha in alphas:
        if alpha == 0.0:
            # Baseline without steering
            inputs = tokenizer(prompt, return_tensors="pt").to(
                next(model.parameters()).device
            )
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=100, 
                                           temperature=0.7, do_sample=True)
            text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        else:
            text = steerer.steer_and_generate(
                prompt, concept_vector, layer_idx, alpha=alpha
            )
        
        results.append((alpha, text))
    
    return results
```

### 5.2 Pattern: Activation Patching (Causal Tracing)

Activation patching identifies which layers are **causally responsible** for a specific behavior by replacing one layer's activations with those from a different run.

```python
def activation_patching(model, tokenizer, clean_text, corrupted_text,
                        layer_range=None, metric="logit_diff"):
    """
    Measure the causal importance of each layer for producing a specific output.
    
    Method:
    1. Run the model on clean_text → record all hidden states and final logits
    2. Run the model on corrupted_text → record final logits as baseline
    3. For each layer L:
       a. Run corrupted_text BUT replace layer L's output with clean_text's layer L output
       b. Measure how much the final logits shift toward the clean output
    
    A layer that "restores" the clean output when patched is causally important.
    
    Args:
        clean_text: text that produces the desired output
        corrupted_text: text that produces a different output (differs minimally from clean)
        layer_range: range of layers to test (default: all)
        metric: "logit_diff" (change in top-1 logit) or "kl_div" or "cosine"
    
    Returns:
        importance: dict mapping layer_idx → float (restoration score)
    
    CRITICAL: clean_text and corrupted_text must tokenize to the SAME number of tokens,
    or you must handle the alignment carefully. For safety, verify:
    """
    device = next(model.parameters()).device
    layers = get_transformer_layers(model)
    num_layers = len(layers)
    
    if layer_range is None:
        layer_range = range(num_layers)
    
    # Step 1: Get clean hidden states and logits
    clean_inputs = tokenizer(clean_text, return_tensors="pt").to(device)
    with torch.no_grad():
        clean_outputs = model(**clean_inputs, output_hidden_states=True)
    clean_logits = clean_outputs.logits[0, -1, :].cpu()
    clean_hidden = {i: clean_outputs.hidden_states[i+1].clone() 
                    for i in layer_range}  # i+1 because hidden_states[0] = embedding
    del clean_outputs
    torch.cuda.empty_cache()
    
    # Step 2: Get corrupted baseline logits
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt").to(device)
    with torch.no_grad():
        corrupted_outputs = model(**corrupted_inputs, output_hidden_states=True)
    corrupted_logits = corrupted_outputs.logits[0, -1, :].cpu()
    del corrupted_outputs
    torch.cuda.empty_cache()
    
    # Step 3: Patch each layer and measure restoration
    importance = {}
    
    for layer_idx in layer_range:
        clean_h = clean_hidden[layer_idx].to(device)
        
        def make_patch_hook(clean_activation):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    patched = clean_activation.clone()
                    return (patched,) + output[1:]
                return clean_activation.clone()
            return hook_fn
        
        hook = layers[layer_idx].register_forward_hook(
            make_patch_hook(clean_h)
        )
        
        with torch.no_grad():
            patched_outputs = model(**corrupted_inputs)
        patched_logits = patched_outputs.logits[0, -1, :].cpu()
        
        hook.remove()
        del patched_outputs
        torch.cuda.empty_cache()
        
        # Compute restoration metric
        if metric == "logit_diff":
            # How much closer are patched logits to clean vs corrupted?
            clean_dist = torch.norm(patched_logits - clean_logits).item()
            corrupt_dist = torch.norm(corrupted_logits - clean_logits).item()
            # 1.0 = fully restored, 0.0 = no restoration
            importance[layer_idx] = 1.0 - (clean_dist / (corrupt_dist + 1e-8))
        elif metric == "cosine":
            cos = torch.nn.functional.cosine_similarity(
                patched_logits.unsqueeze(0), clean_logits.unsqueeze(0)
            ).item()
            importance[layer_idx] = cos
        
        clean_h.cpu()
    
    return importance
```

### 5.3 Pattern: Multi-Vector Injection (Multiple Concepts Simultaneously)

For NeuroOS-style multi-channel injection, where different types of information
are injected at different depths simultaneously.

```python
class MultiChannelInjector:
    """
    Inject different concept vectors at different layers simultaneously.
    
    This is the closest software approximation to the NeuroOS Depth Injection Router.
    Each "channel" injects a vector at a specified layer with a specified strength.
    
    Usage:
        injector = MultiChannelInjector(model, tokenizer)
        injector.add_channel("temperature", temp_vector, layer=28, alpha=3.0)
        injector.add_channel("speed", speed_vector, layer=22, alpha=2.0)
        injector.add_channel("emotion", emotion_vector, layer=18, alpha=1.5)
        
        result = injector.generate("Describe the scene: ")
        injector.clear_channels()
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_transformer_layers(model)
        self.channels = {}
        self._hooks = []
    
    def add_channel(self, name, vector, layer, alpha=1.0):
        self.channels[name] = {
            "vector": vector,
            "layer": layer,
            "alpha": alpha,
        }
    
    def remove_channel(self, name):
        del self.channels[name]
    
    def clear_channels(self):
        self.channels.clear()
    
    def _attach_hooks(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        # Group channels by layer for efficiency
        layer_to_channels = {}
        for name, ch in self.channels.items():
            layer_idx = ch["layer"]
            if layer_idx not in layer_to_channels:
                layer_to_channels[layer_idx] = []
            layer_to_channels[layer_idx].append(ch)
        
        for layer_idx, channels in layer_to_channels.items():
            # Pre-compute combined steering vector for this layer
            combined = torch.zeros(self.model.config.hidden_size, dtype=dtype, device=device)
            for ch in channels:
                v = torch.tensor(ch["vector"], dtype=dtype, device=device)
                combined += ch["alpha"] * v
            
            def make_hook(steering):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        return (output[0] + steering,) + output[1:]
                    return output + steering
                return hook_fn
            
            hook = self.layers[layer_idx].register_forward_hook(make_hook(combined))
            self._hooks.append(hook)
    
    def _detach_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.7):
        self._attach_hooks()
        
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        
        self._detach_hooks()
        
        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
```

---

## Part 6 — The Concept Library: Storage and Retrieval

### 6.1 Data Structure for Concept Vectors

```python
import json
import os

class ConceptLibrary:
    """
    Persistent storage and retrieval system for concept vectors.
    
    Each concept entry stores:
        - name: human-readable concept identifier
        - vector: the concept direction (numpy array)
        - layer_idx: the layer it was extracted from
        - optimal_layer: the layer where probe accuracy is highest
        - accuracy_by_layer: dict mapping layer → probe accuracy
        - metadata: extraction parameters, source texts, timestamps
    
    File format: .npz (numpy compressed) for vectors + .json for metadata
    """
    
    def __init__(self, library_dir="concept_library"):
        self.library_dir = library_dir
        os.makedirs(library_dir, exist_ok=True)
        self.metadata_file = os.path.join(library_dir, "catalog.json")
        self.catalog = self._load_catalog()
    
    def _load_catalog(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_catalog(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.catalog, f, indent=2)
    
    def add_concept(self, name, vector, layer_idx, accuracy_by_layer=None,
                    positive_texts=None, negative_texts=None, model_name=None):
        """Store a concept vector with full metadata."""
        # Save vector
        vec_file = os.path.join(self.library_dir, f"{name}.npy")
        np.save(vec_file, vector)
        
        # Save per-layer accuracy profile if available
        if accuracy_by_layer is not None:
            profile_file = os.path.join(self.library_dir, f"{name}_profile.npy")
            layers = sorted(accuracy_by_layer.keys())
            accs = [accuracy_by_layer[l] for l in layers]
            np.save(profile_file, np.array(accs))
        
        # Update catalog
        optimal_layer = layer_idx
        if accuracy_by_layer:
            optimal_layer = max(accuracy_by_layer, key=accuracy_by_layer.get)
        
        self.catalog[name] = {
            "layer_idx": layer_idx,
            "optimal_layer": optimal_layer,
            "hidden_dim": len(vector),
            "model_name": model_name,
            "accuracy_by_layer": {str(k): v for k, v in (accuracy_by_layer or {}).items()},
            "num_positive_texts": len(positive_texts) if positive_texts else 0,
            "num_negative_texts": len(negative_texts) if negative_texts else 0,
        }
        self._save_catalog()
    
    def get_concept(self, name):
        """Retrieve a concept vector and its metadata."""
        vec_file = os.path.join(self.library_dir, f"{name}.npy")
        vector = np.load(vec_file)
        metadata = self.catalog.get(name, {})
        return vector, metadata
    
    def get_all_vectors_at_layer(self, layer_idx):
        """Get all concept vectors that are optimal at a given layer."""
        result = {}
        for name, meta in self.catalog.items():
            if meta.get("optimal_layer") == layer_idx:
                vec, _ = self.get_concept(name)
                result[name] = vec
        return result
    
    def project_text(self, model, tokenizer, text, layer_idx=None):
        """Project a text onto all concepts in the library."""
        profile = {}
        for name, meta in self.catalog.items():
            use_layer = layer_idx or meta.get("optimal_layer", meta.get("layer_idx"))
            vec, _ = self.get_concept(name)
            vec_unit = vec / np.linalg.norm(vec)
            
            text_vec = extract_at_layer(model, tokenizer, [text], use_layer)[0]
            profile[name] = float(np.dot(text_vec, vec_unit))
        
        return profile
    
    def list_concepts(self):
        """List all stored concepts with summary info."""
        for name, meta in sorted(self.catalog.items()):
            opt_layer = meta.get("optimal_layer", "?")
            dim = meta.get("hidden_dim", "?")
            acc = meta.get("accuracy_by_layer", {}).get(str(opt_layer), "?")
            print(f"  {name:30s}  optimal_layer={opt_layer:>3}  "
                  f"dim={dim}  accuracy={acc}")
    
    def get_depth_map(self):
        """
        Return the depth-semantics map: which concepts peak at which layers.
        This is the core data structure for NeuroOS Depth Injection Router.
        """
        depth_map = {}
        for name, meta in self.catalog.items():
            opt_layer = meta.get("optimal_layer")
            if opt_layer not in depth_map:
                depth_map[opt_layer] = []
            depth_map[opt_layer].append(name)
        return dict(sorted(depth_map.items()))
```

### 6.2 Predefined Concept Taxonomies

When generating probing datasets, use this taxonomy as a guide. Each level
corresponds to an expected depth range in the transformer.

```python
CONCEPT_TAXONOMY = {
    # ═══════════════════════════════════════════════════════════════
    # LEVEL 1 — SURFACE (expected peak: layers 0–15% of total)
    # Linguistic features, token-level properties
    # ═══════════════════════════════════════════════════════════════
    "surface": {
        "singular_vs_plural": {
            "description": "Grammatical number",
            "positive": [  # singular
                "The cat sits on the mat",
                "A single star shines in the sky",
                # ... minimum 15 examples, diversified
            ],
            "negative": [  # plural
                "The cats sit on the mats",
                "Many stars shine in the sky",
            ],
        },
        "past_vs_present_tense": {
            "description": "Verb tense",
            # ...
        },
        "formal_vs_informal": {
            "description": "Register",
            # ...
        },
    },
    
    # ═══════════════════════════════════════════════════════════════
    # LEVEL 2 — SYNTACTIC (expected peak: layers 15–40%)
    # Structural properties, grammatical relations
    # ═══════════════════════════════════════════════════════════════
    "syntactic": {
        "active_vs_passive_voice": {
            "description": "Sentence voice",
            # ...
        },
        "simple_vs_complex_sentence": {
            "description": "Syntactic complexity",
            # ...
        },
        "question_vs_statement": {
            "description": "Sentence type",
            # ...
        },
    },
    
    # ═══════════════════════════════════════════════════════════════
    # LEVEL 3 — SEMANTIC (expected peak: layers 40–70%)
    # Conceptual categories, entity types, abstract properties
    # ═══════════════════════════════════════════════════════════════
    "semantic": {
        "animate_vs_inanimate": {
            "description": "Whether the subject is a living entity",
            # ...
        },
        "concrete_vs_abstract": {
            "description": "Concreteness of the main concept",
            # ...
        },
        "positive_vs_negative_sentiment": {
            "description": "Emotional valence",
            # ...
        },
        "spatial_near_vs_far": {
            "description": "Spatial proximity",
            # ...
        },
    },
    
    # ═══════════════════════════════════════════════════════════════
    # LEVEL 4 — DEEP KNOWLEDGE (expected peak: layers 70–90%)
    # World knowledge, physical properties, causal reasoning
    # ═══════════════════════════════════════════════════════════════
    "deep_knowledge": {
        "hot_vs_cold": {
            "description": "Temperature (physical property)",
            # ...
        },
        "heavy_vs_light": {
            "description": "Mass/weight (physical property)",
            # ...
        },
        "fast_vs_slow": {
            "description": "Speed/velocity (physical property)",
            # ...
        },
        "cause_vs_effect": {
            "description": "Causal direction",
            # ...
        },
        "dangerous_vs_safe": {
            "description": "Risk assessment (world knowledge)",
            # ...
        },
    },
    
    # ═══════════════════════════════════════════════════════════════
    # LEVEL 5 — PRE-LINGUISTIC (expected peak: layers 85–95%)
    # Geometric, physical, mathematical structure
    # ═══════════════════════════════════════════════════════════════
    "prelinguistic": {
        "motion_vs_stasis": {
            "description": "Dynamic vs static (physical state)",
            # ...
        },
        "containment_vs_exposure": {
            "description": "Inside vs outside (spatial topology)",
            # ...
        },
        "increasing_vs_decreasing": {
            "description": "Monotonic trend direction",
            # ...
        },
        "symmetry_vs_asymmetry": {
            "description": "Structural symmetry",
            # ...
        },
    },
}
```

**Rules for generating contrastive text pairs:**

1. **Minimal pair principle**: The two sets should differ ONLY in the target concept. All other features (topic, complexity, style, length) should be matched as closely as possible.

2. **Lexical diversity**: Do not repeat the same key word across examples. If probing for "hot", do NOT include the word "hot" in every sentence. Use descriptions, implications, scenarios — not labels.

3. **Topic diversity**: Spread examples across domains. For "hot vs cold", use cooking, weather, physics, geography, industrial processes, human body sensations — not just one domain.

4. **Avoid confounds**: If all "hot" sentences mention fire and all "cold" sentences mention ice, you might be finding the fire/ice vector, not the temperature vector. Ensure the concept is the ONLY systematic difference.

5. **Balance lengths**: If positive texts average 12 tokens and negative texts average 6 tokens, the probe might learn length instead of the concept.

6. **Minimum 15 examples per class** for reliable probing, 30+ for stable concept vectors.

---

## Part 7 — Sparse Autoencoder (SAE) Decomposition

### 7.1 What SAEs Do and Why They Matter

A Sparse Autoencoder decomposes a hidden state vector (e.g., dimension 4096) into
a much larger but SPARSE representation (e.g., dimension 65536) where each dimension
corresponds to an individual, interpretable feature.

```
Hidden state (4096 dims, dense, polysemantic)
    │
    ▼
SAE Encoder: W_enc @ h + b_enc → ReLU → sparse activations
    │         (65536 dims, >99% zeros, monosemantic)
    │
    ▼
SAE Decoder: W_dec @ sparse_activations + b_dec → reconstructed hidden state
    │         (4096 dims, dense, approximates original)
```

The key insight: in the sparse representation, each active dimension (feature)
corresponds to roughly ONE concept. Feature 4721 might always activate for
mentions of water. Feature 12003 might always activate for father-child
relationships. This resolves the superposition problem.

### 7.2 Training a Simple SAE on Extracted Activations

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for decomposing hidden states into interpretable features.
    
    Architecture:
        Encoder: hidden_dim → expansion_factor * hidden_dim (with ReLU)
        Decoder: expansion_factor * hidden_dim → hidden_dim (with unit-norm columns)
    
    Loss: reconstruction_loss + l1_coefficient * sparsity_loss
    """
    
    def __init__(self, hidden_dim, expansion_factor=16, l1_coefficient=1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dict_size = hidden_dim * expansion_factor
        self.l1_coefficient = l1_coefficient
        
        # Encoder
        self.encoder = nn.Linear(hidden_dim, self.dict_size)
        
        # Decoder (no bias — reconstructs centered activations)
        self.decoder = nn.Linear(self.dict_size, hidden_dim, bias=True)
        
        # Initialize decoder columns as unit vectors
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def encode(self, x):
        """Encode hidden states into sparse feature activations."""
        return torch.relu(self.encoder(x))
    
    def decode(self, features):
        """Decode sparse features back to hidden state space."""
        return self.decoder(features)
    
    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def loss(self, x):
        reconstruction, features = self.forward(x)
        recon_loss = ((x - reconstruction) ** 2).mean()
        sparsity_loss = features.abs().mean()
        total_loss = recon_loss + self.l1_coefficient * sparsity_loss
        return total_loss, {
            "reconstruction": recon_loss.item(),
            "sparsity": sparsity_loss.item(),
            "total": total_loss.item(),
            "active_features": (features > 0).float().mean().item(),
        }


def collect_training_data(model, tokenizer, texts, layer_idx, batch_size=4):
    """
    Collect hidden states for SAE training.
    
    Collects ALL token positions (not just last) since SAEs should learn
    a general decomposition of the representation space.
    
    Returns:
        data: numpy array of shape (total_tokens, hidden_dim)
    """
    all_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[layer_idx]  # (batch, seq, hidden)
        mask = inputs["attention_mask"]             # (batch, seq)
        
        # Extract all non-padding positions
        for b in range(hidden.shape[0]):
            valid_len = mask[b].sum().item()
            all_vectors.append(hidden[b, :valid_len, :].cpu().numpy())
        
        del outputs
        torch.cuda.empty_cache()
    
    return np.vstack(all_vectors)


def train_sae(hidden_data, hidden_dim, expansion_factor=16, 
              l1_coefficient=1e-3, epochs=10, batch_size=256, lr=3e-4):
    """
    Train a Sparse Autoencoder on collected hidden states.
    
    Args:
        hidden_data: numpy array of shape (N, hidden_dim)
        hidden_dim: dimension of hidden states
        expansion_factor: how many times larger the SAE dictionary is
        l1_coefficient: sparsity penalty weight
            - Too high → features too sparse, poor reconstruction
            - Too low → features not sparse enough, hard to interpret
            - Start with 1e-3, adjust based on active_features percentage
            - Target: 1-5% of features active per input
    
    Returns:
        sae: trained SparseAutoencoder model
        training_log: list of per-epoch metrics
    """
    sae = SparseAutoencoder(hidden_dim, expansion_factor, l1_coefficient)
    sae = sae.cuda() if torch.cuda.is_available() else sae
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    dataset = torch.tensor(hidden_data, dtype=torch.float32)
    
    training_log = []
    
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(dataset))
        epoch_losses = []
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[perm[i:i+batch_size]]
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            loss, metrics = sae.loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Re-normalize decoder columns
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0
                )
            
            epoch_losses.append(metrics)
        
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_losses])
            for k in epoch_losses[0]
        }
        training_log.append(avg_metrics)
        print(f"Epoch {epoch+1}/{epochs}: "
              f"recon={avg_metrics['reconstruction']:.4f}, "
              f"sparsity={avg_metrics['sparsity']:.4f}, "
              f"active={avg_metrics['active_features']:.3%}")
    
    return sae, training_log
```

### 7.3 Interpreting SAE Features

```python
def find_max_activating_examples(sae, model, tokenizer, texts, layer_idx,
                                  feature_idx, top_k=10):
    """
    Find which texts maximally activate a specific SAE feature.
    This is the primary method for understanding what a feature represents.
    """
    activations = []
    
    for text in texts:
        vec = extract_at_layer(model, tokenizer, [text], layer_idx)[0]
        vec_tensor = torch.tensor(vec, dtype=torch.float32)
        if torch.cuda.is_available():
            vec_tensor = vec_tensor.cuda()
        
        features = sae.encode(vec_tensor.unsqueeze(0))
        activation = features[0, feature_idx].item()
        activations.append((activation, text))
    
    # Sort by activation strength
    activations.sort(key=lambda x: x[0], reverse=True)
    
    return activations[:top_k]


def feature_concept_alignment(sae, concept_vector, top_k=10):
    """
    Find which SAE features align best with a known concept vector.
    
    This bridges the two approaches: contrastive probing finds directions,
    SAE finds features. This function maps between them.
    """
    # Each column of the decoder weight matrix is a feature direction
    decoder_weights = sae.decoder.weight.data.cpu().numpy()  # (hidden_dim, dict_size)
    
    concept_unit = concept_vector / np.linalg.norm(concept_vector)
    
    # Compute alignment of each feature with the concept
    alignments = decoder_weights.T @ concept_unit  # (dict_size,)
    
    # Top aligned features
    top_indices = np.argsort(np.abs(alignments))[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "feature_idx": int(idx),
            "alignment": float(alignments[idx]),
            "direction": "positive" if alignments[idx] > 0 else "negative",
        })
    
    return results
```

---

## Part 8 — Complete Experimental Workflows

### 8.1 Workflow: Build a Depth-Semantics Map

```python
def build_depth_semantics_map(model, tokenizer, concept_taxonomy, 
                               output_dir="depth_map_results"):
    """
    Complete workflow to map which concepts are encoded at which layers.
    
    This is the foundational experiment for NeuroOS Depth Injection Router.
    
    Args:
        concept_taxonomy: dict structured like CONCEPT_TAXONOMY above
        output_dir: where to save results
    
    Produces:
        - Per-concept accuracy-vs-layer curves
        - Depth-semantics map (concept → optimal layer)
        - Concept vector library
        - Summary visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    library = ConceptLibrary(os.path.join(output_dir, "concept_library"))
    
    all_results = {}
    
    for level_name, concepts in concept_taxonomy.items():
        for concept_name, concept_data in concepts.items():
            full_name = f"{level_name}/{concept_name}"
            print(f"\nProbing: {full_name}")
            
            pos_texts = concept_data["positive"]
            neg_texts = concept_data["negative"]
            
            # Probe across all layers
            results = probe_across_all_layers(
                model, tokenizer, pos_texts, neg_texts
            )
            
            # Find optimal layer
            layer_accuracies = {l: r["accuracy"] for l, r in results.items()}
            optimal_layer = max(layer_accuracies, key=layer_accuracies.get)
            
            # Store concept vector from optimal layer
            concept_vec = results[optimal_layer]["concept_vector"]
            if concept_vec is not None:
                library.add_concept(
                    name=full_name,
                    vector=concept_vec,
                    layer_idx=optimal_layer,
                    accuracy_by_layer=layer_accuracies,
                    positive_texts=pos_texts,
                    negative_texts=neg_texts,
                    model_name=model.config._name_or_path,
                )
            
            all_results[full_name] = {
                "level": level_name,
                "optimal_layer": optimal_layer,
                "peak_accuracy": layer_accuracies[optimal_layer],
                "accuracy_curve": layer_accuracies,
            }
            
            print(f"  → Optimal layer: {optimal_layer}, "
                  f"Peak accuracy: {layer_accuracies[optimal_layer]:.3f}")
    
    # Save summary
    with open(os.path.join(output_dir, "depth_map_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print depth map
    print("\n" + "=" * 60)
    print("DEPTH-SEMANTICS MAP")
    print("=" * 60)
    depth_map = library.get_depth_map()
    for layer, concepts in depth_map.items():
        print(f"  Layer {layer:3d}: {', '.join(concepts)}")
    
    return all_results, library
```

### 8.2 Workflow: Concept-Steered Generation Benchmark

```python
def benchmark_steering(model, tokenizer, library, test_prompts, 
                       concept_name, alphas=None):
    """
    Systematically test concept steering across alpha values and layers.
    
    Produces a grid of (alpha × layer) showing steering effectiveness.
    """
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    
    concept_vec, meta = library.get_concept(concept_name)
    optimal_layer = meta["optimal_layer"]
    
    # Test at optimal layer and nearby layers
    test_layers = [
        max(0, optimal_layer - 10),
        max(0, optimal_layer - 5),
        optimal_layer,
        min(model.config.num_hidden_layers - 1, optimal_layer + 5),
        min(model.config.num_hidden_layers - 1, optimal_layer + 10),
    ]
    test_layers = sorted(set(test_layers))
    
    results = {}
    steerer = ActivationSteerer(model, tokenizer)
    
    for layer_idx in test_layers:
        results[layer_idx] = {}
        for alpha in alphas:
            generations = []
            for prompt in test_prompts:
                if alpha == 0.0:
                    inputs = tokenizer(prompt, return_tensors="pt").to(
                        next(model.parameters()).device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=60,
                                           temperature=0.7, do_sample=True)
                    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
                else:
                    text = steerer.steer_and_generate(
                        prompt, concept_vec, layer_idx, alpha=alpha,
                        max_new_tokens=60
                    )
                generations.append(text)
            
            results[layer_idx][alpha] = generations
    
    return results
```

---

## Part 9 — Error Handling and Diagnostics

### 9.1 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| All probe accuracies ≈ 0.50 | Concept not encoded OR texts too similar between classes | Verify texts differ clearly on the target concept; increase dataset size |
| Probe accuracy > 0.95 with <20 samples | Overfitting or confound | Reduce C parameter; check for length/lexical confounds |
| OOM during extraction | Hidden states too large | Reduce batch_size; use 8-bit quantization; extract fewer layers |
| Steering has no visible effect | Alpha too low OR wrong layer | Increase alpha; try layer ± 5 from optimal |
| Steering produces gibberish | Alpha too high OR wrong layer type | Reduce alpha; ensure you are hooking the full layer, not a sublayer |
| Cosine similarity all ≈ 1.0 | Comparing hidden states with huge norms; residual stream dominance | Center the vectors (subtract mean) before computing similarity |
| SAE features all look similar | L1 coefficient too low; not enough training data | Increase l1_coefficient; collect more diverse training examples |

### 9.2 Sanity Checks

```python
def run_sanity_checks(model, tokenizer):
    """Run basic checks to verify the probing setup is working correctly."""
    
    # Check 1: hidden_states are accessible
    text = "Hello, world."
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    assert out.hidden_states is not None, "output_hidden_states not enabled"
    n_layers = len(out.hidden_states) - 1
    hidden_dim = out.hidden_states[0].shape[-1]
    print(f"✓ Hidden states accessible: {n_layers} layers, dim={hidden_dim}")
    
    # Check 2: hidden states change across layers
    first = out.hidden_states[1][0, -1, :].cpu()
    last = out.hidden_states[-1][0, -1, :].cpu()
    cos = torch.nn.functional.cosine_similarity(
        first.unsqueeze(0), last.unsqueeze(0)
    ).item()
    assert cos < 0.99, f"First and last layer too similar ({cos:.4f})"
    print(f"✓ Layer representations differ: cos(layer1, layer{n_layers}) = {cos:.3f}")
    
    # Check 3: different texts produce different representations
    text2 = "The quick brown fox jumps over the lazy dog."
    inputs2 = tokenizer(text2, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out2 = model(**inputs2, output_hidden_states=True)
    v1 = out.hidden_states[n_layers//2][0, -1, :].cpu()
    v2 = out2.hidden_states[n_layers//2][0, -1, :].cpu()
    cos2 = torch.nn.functional.cosine_similarity(
        v1.unsqueeze(0), v2.unsqueeze(0)
    ).item()
    assert cos2 < 0.99, f"Different texts too similar ({cos2:.4f})"
    print(f"✓ Different texts produce different vectors: cos = {cos2:.3f}")
    
    # Check 4: VRAM status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"✓ VRAM: {allocated:.1f} / {total:.1f} GB "
              f"({allocated/total:.0%} used)")
    
    del out, out2
    torch.cuda.empty_cache()
    print("✓ All sanity checks passed")
```

---

## Part 10 — Glossary for Code Generators

When the user requests code involving these terms, this is what they mean:

| User Term | Technical Meaning | Key Variable |
|---|---|---|
| "hidden state" / "activation" | Output tensor of a transformer layer | `outputs.hidden_states[layer_idx]` |
| "concept vector" / "direction" | Unit vector in hidden space representing a concept | numpy array, shape `(hidden_dim,)` |
| "probing" / "probe" | Training a linear classifier on hidden states | `LogisticRegression` or `Ridge` |
| "steering" / "injection" | Adding a scaled vector to hidden states during generation | Hook-based modification |
| "patching" | Replacing one run's hidden states with another's | Hook-based substitution |
| "cosine similarity" | Dot product of unit vectors; measures direction alignment | `sklearn.metrics.pairwise.cosine_similarity` |
| "UMAP" / "dimensionality reduction" | Projecting high-dim vectors to 2D/3D for visualization | `umap.UMAP` or `sklearn.decomposition.PCA` |
| "SAE" / "sparse autoencoder" | Neural net that decomposes dense vectors into sparse interpretable features | Custom `nn.Module` |
| "depth map" | Mapping from concept types to optimal transformer layers | Dict: `{layer_idx: [concept_names]}` |
| "residual stream" | The main data pathway through the transformer (hidden states + skip connections) | `hidden_states[i]` |
| "attention weights" / "attention pattern" | The softmax matrix showing which tokens attend to which | `outputs.attentions[layer_idx]` |
| "logits" | Raw (pre-softmax) scores over vocabulary for next token prediction | `outputs.logits` |
| "feature" (in SAE context) | A single interpretable direction learned by the sparse autoencoder | Column of SAE decoder weight matrix |
| "polysemantic" | A single neuron/dimension encodes multiple unrelated concepts | The problem SAEs solve |
| "monosemantic" | A single feature encodes exactly one concept | The goal of SAEs |
| "superposition" | Multiple concepts encoded in overlapping directions in fewer dimensions | Why individual neurons are hard to interpret |

---

## Appendix A — Hardware-Specific Notes

### AMD MI50 (ROCm)

```bash
# Verify ROCm installation
rocm-smi                               # GPU status
rocminfo | grep "Name:"                # Device name
python -c "import torch; print(torch.cuda.is_available())"  # Should print True

# PyTorch uses the same torch.cuda API for ROCm
# torch.cuda.empty_cache() works on ROCm
# torch.cuda.memory_allocated() works on ROCm

# Known issues:
# - bitsandbytes may require ROCm-specific build
# - flash-attention may need ROCm-compatible version
# - Some custom CUDA kernels won't work — stick to pure PyTorch operations
```

### Memory Management for 32GB VRAM

```python
# Rule of thumb for MI50 32GB:
# 12B model in FP16 = ~24GB → 8GB free for operations
# 12B model in INT8 = ~12GB → 20GB free for operations

# Best practices:
# 1. Always use torch.no_grad() for extraction
# 2. Move tensors to CPU immediately after extraction
# 3. Delete GPU tensors and call torch.cuda.empty_cache()
# 4. Process in small batches (batch_size=2-4)
# 5. Don't extract all layers at once — do one layer at a time if needed
# 6. For SAE training, collect data to CPU first, then train SAE on GPU separately

# Monitor VRAM during experiments:
def print_vram():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {used:.1f}GB used, {cached:.1f}GB cached, {total:.1f}GB total")
```

---

*This document is a technical reference for AI code generation agents. It should be loaded into the context of any coding session involving LLM internal representation analysis, concept vector extraction, activation steering, or interpretability research on transformer-based language models.*
