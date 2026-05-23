# AGENTS.md — AI Agent Development Guide for OpenMythos
## Project Overview
**OpenMythos** is an open-source theoretical implementation of the Claude Mythos **Recurrent-Depth Transformer (RDT)** architecture. The project is split into two layers:
- **`open_mythos`**: Core transformer model (PyTorch) with three-stage architecture and configurable attention mechanisms
- **`mythos_glasseye`**: Higher-level agent-driven workflow system (pipeline runner, CLI, Gemini integration)
---
## The Three-Stage Architecture — The "Why"
The model is fundamentally different from standard transformers. Understand this structure before any modification:
```
Input IDs
  ↓
[Prelude]          — Standard transformer blocks, run ONCE
  ↓
[Recurrent Block]  — Looped up to T times (default: T=16)
  ↑_________↓      — h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
  ↓
[Coda]             — Standard transformer blocks, run ONCE
  ↓
Output logits
```
Key insight: **The hidden state recirculates through the same block T times, with input injection `e` at each step.** This is not stacking layers; it's recycling them.
### Why This Matters for Devs
1. **MoE FFN lives ONLY in the Recurrent Block** — prelude/coda use dense SwiGLU. This is deliberate.
2. **RoPE and KV cache behave differently per attention type** (GQA vs MLA).
3. **Loop stability is enforced by construction** — the injection matrix A is parameterized as `Diag(-exp(log_A))` so that spectral radius ρ(A) < 1 always holds, preventing training divergence.
4. **Each loop can compute differently** — thanks to loop-index positional embedding, the same weights behave differently at depth 0 vs depth 15.
---
## Configuration via `MythosConfig`
All model behavior is controlled by a single dataclass (`MythosConfig`). **Never hardcode dimensions.**
### Core Parameters (Always Matter)
```python
MythosConfig(
    vocab_size=32000,           # Token vocabulary size
    dim=2048,                   # Hidden dimension (model width)
    n_heads=16,                 # Query attention heads
    max_seq_len=4096,           # RoPE precomputation length
    max_loop_iters=16,          # Default recurrent depth T
    prelude_layers=2,           # Blocks before loop
    coda_layers=2,              # Blocks after loop
    attn_type="mla",            # "mla" or "gqa"
)
```
### Attention-Type Dispatch (Critical Pattern)
The codebase switches attention mechanisms via `cfg.attn_type`:
```python
if cfg.attn_type == "gqa":
    # Use GQAttention (fewer KV heads)
    # Relevant: n_kv_heads, rope_theta
    cfg = MythosConfig(n_heads=16, n_kv_heads=4, attn_type="gqa")
else:  # "mla"
    # Use MLAttention (compressed KV latent)
    # Relevant: kv_lora_rank, q_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim
    cfg = MythosConfig(
        n_heads=16, attn_type="mla",
        kv_lora_rank=512, q_lora_rank=1536,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128
    )
```
**Pattern**: When touching attention code, check `cfg.attn_type` and apply changes to both `GQAttention` and `MLAttention` consistently.
### Pre-Configured Variants
Import from `open_mythos.variants`:
```python
from open_mythos import mythos_3b, mythos_1b, mythos_10b
cfg = mythos_3b()  # Returns pre-configured MythosConfig
model = OpenMythos(cfg)
```
Available: `mythos_1b`, `mythos_3b`, `mythos_10b`, `mythos_50b`, `mythos_100b`, `mythos_500b`, `mythos_1t`.
---
## Critical Architectural Components
### 1. **LTIInjection** (The Stability Guarantor)
Located in `open_mythos/main.py`. This ensures the recurrent loop doesn't explode.
```python
class LTIInjection(nn.Module):
    """
    Enforces stability for the recurrent update:
    h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
    A is parameterized as Diag(-exp(log_A)) so ρ(A) < 1 always.
    """
    def get_A(self) -> torch.Tensor:
        # Returns the discrete injection matrix (diagonal)
        # Spectral radius check: A.abs().max() < 1.0
```
**Core invariant**: If you modify injection parameters, guarantee `ρ(A) < 1`. Test with:
```python
A = model.recurrent.injection.get_A()
assert (A.abs() < 1.0).all(), "Spectral radius violation!"
```
### 2. **MoEFFN** (Recurrent Block Only)
```python
class MoEFFN(nn.Module):
    """Sparse mixture-of-experts FFN with routed + shared experts."""
    # Config fields:
    # - n_experts: Total routed experts
    # - n_shared_experts: Always-active shared experts
    # - n_experts_per_tok: Top-K routed per token
    # - expert_dim: Hidden size inside each expert
```
**Important**: MoE is ONLY in the Recurrent Block. The Prelude and Coda use dense SwiGLU:
```python
class TransformerBlock(nn.Module):
    """Standard transformer with dense SwiGLU or MoE FFN."""
    # If used in recurrent block: FFN is MoEFFN
    # If used in prelude/coda: FFN is dense SwiGLU
```
### 3. **ACTHalting** (Adaptive Computation Time)
```python
class ACTHalting(nn.Module):
    """Per-position early exit: loops can halt when halting.cumsum() > act_threshold."""
```
This allows variable compute per token/sequence. Left as a framework here; production likely has sophisticated gating.
### 4. **LoRAAdapter** (Depth Adaptation)
```python
class LoRAAdapter(nn.Module):
    """Depth-wise LoRA: adjusts recurrent block behavior at each loop iteration."""
    # Low-rank matrices apply per-depth adjustments
    # Same weights, different effective computation at each depth t
```
---
## Testing Pattern (test_main.py)
Small, deterministic configs for CPU-based tests:
```python
def gqa_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=200,      # Tiny for fast tests
        dim=64,
        n_heads=4, n_kv_heads=2,
        max_loop_iters=3,     # Short loops in tests
        attn_type="gqa",
        # ... other fields
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)
```
**Pattern**: Always test against both `gqa_cfg()` and `mla_cfg()` configs. Component tests verify shape, not numerical correctness:
```python
def test_forward_shape(self):
    cfg = gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(ids, n_loops=3)
    assert logits.shape == (2, 8, cfg.vocab_size)
```
---
## Training Workflow (training/3b_fine_web_edu.py)
Reference implementation for pretraining:
```bash
# Single GPU
python training/3b_fine_web_edu.py
# Multi-GPU (auto-detects)
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```
### Key Design Choices (DONT CHANGE WITHOUT REASON)
| Component | Choice | Why |
|-----------|--------|-----|
| **Parallelism** | FSDP (Fully Sharded Data Parallel) | Scales training across GPUs/TPUs |
| **Dataset** | `HuggingFaceFW/fineweb-edu` (streaming) | Billions of tokens, no disk materialization |
| **Tokenizer** | `openai/gpt-oss-20b` via `MythosTokenizer` | GPT-2 compatible |
| **Precision** | bfloat16 (H100/A100) or float16 + GradScaler | Stability + memory efficiency |
| **Schedule** | Linear warmup (2000 steps) → cosine decay | Standard for pretraining |
| **Target tokens** | ~30B (Chinchilla-adjusted for looping) | Accounts for loop depth regularization |
### Dataset Pattern
```python
class FineWebEduDataset(IterableDataset):
    """Streaming dataset with 2D sharding: (rank, worker_id) → 1 shard each."""
```
- Uses `world_size × num_workers` shards
- Each process deterministically owns one shard
- No cross-process coordination needed
- Resumable from beginning (acceptable for pretraining cost)
---
## CLI Tools & Agent System
### `openmythos` CLI (open_mythos/cli.py)
```bash
openmythos generate --model 3b --prompt "Once upon a time" --max-tokens 100
openmythos benchmark --size 3b --device cuda
```
Dynamic model selection + config building via inspection. See `_filtered_kwargs` pattern for flexible config passing.
### `glasseye` CLI (mythos_glasseye/cli.py)
Higher-level orchestration:
```bash
glasseye build --size tiny --device cpu
glasseye train --steps 1000 --batch-size 4
glasseye serve --model artifacts/model.pt --port 5000
glasseye list-agents
```
### Agent Registry Pattern (mythos_glasseye/registry.py)
```python
registry = AgentRegistry()
builder = registry.get("model_builder")
result = builder.run(model_name="mythos-3b", config={...}, device="cuda")
```
Agents auto-discovered from `mythos_glasseye/agents/` module. Each agent:
- Inherits from `BaseAgent`
- Implements `run(**kwargs) -> AgentResult`
- Returns `AgentResult(success, output_path, metadata)`
---
## YAML Pipeline Execution
Pipelines are in `pipelines/`. Example structure:
```yaml
steps:
  - name: build_checkpoint
    agent: model_builder
    args:
      model_name: "glasseye-3b"
      device: "cuda"
  - name: train
    agent: trainer
    args:
      max_steps: 1000
      batch_size: 4
      # Can interpolate: {{build_checkpoint.output_path}}
```
Executed via:
```python
from mythos_glasseye.pipeline_runner import run_pipeline_file
summary = run_pipeline_file("pipelines/training.yaml")
```
---
## Code Formatting & Linting (ALWAYS RUN BEFORE COMMIT)
```bash
# Format code (line length: 88 characters)
black .
# Lint
ruff check .
# Test
pytest tests/
```
**Black target**: Python 3.10+, line length 88 (set in `pyproject.toml`).
---
## Key Conventions & Patterns
### 1. **Shape Assertions in Forward Passes**
Always include in docstrings:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, T, dim)
    Returns:
        (B, T, dim)
    """
```
### 2. **Config Validation**
`MythosConfig` uses Pydantic (v2). Validation runs automatically on instantiation. Add custom validators if needed:
```python
@field_validator("n_experts")
def validate_experts(cls, v):
    assert v % 2 == 0, "n_experts must be even"
    return v
```
### 3. **Device-Agnostic Code**
Use `x.device` and `x.dtype`, not hardcoded `"cuda"`:
```python
# Good
freqs = precompute_rope_freqs(dim, max_len, device=input_ids.device)
# Bad
freqs = precompute_rope_freqs(dim, max_len, device="cuda:0")
```
### 4. **Buffer Registration**
RoPE frequencies are registered as buffers (not parameters):
```python
self.register_buffer("freqs_cis", precompute_rope_freqs(...))
```
This ensures they move to the correct device/dtype with `model.to()` or `.cuda()`.
### 5. **Import Organization**
Standard order in files:
```python
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_mythos.main import SomeClass
from mythos_glasseye.registry import AgentRegistry
```
---
## Dependencies & Build System
- **Package manager**: Poetry (see `pyproject.toml`)
- **Python**: 3.10+
- **Core deps**: PyTorch 2.11.0, transformers ≥4.40.0, Pydantic ≥2.0
- **Optional**: Flash Attention 2 (flash-attn ≥2.8.3)
Install with:
```bash
pip install -e .                    # Core
pip install -e .[flash]             # With Flash Attention
poetry install                      # Using Poetry
```
---
## Where to Make Changes
| Task | File(s) |
|------|---------|
| Add model variant | `open_mythos/variants.py` |
| Modify attention mechanism | `open_mythos/main.py` (GQAttention, MLAttention classes) |
| Change training loop | `training/3b_fine_web_edu.py` |
| Add CLI command | `open_mythos/cli.py` or `mythos_glasseye/cli.py` |
| Add agent | Create `mythos_glasseye/agents/my_agent.py` (auto-discovered) |
| Add YAML pipeline | Create `pipelines/my_pipeline.yaml` |
| Tokenizer customization | `open_mythos/tokenizer.py` |
| Core architecture | `open_mythos/main.py` (OpenMythos, RecurrentBlock classes) |
---
## Summary: What Makes This Codebase Different
1. **Looping, not stacking** — Recycled weights, not added layers
2. **Stability by construction** — Injection matrix A guaranteed to have ρ(A) < 1
3. **Attention switchability** — Single codebase supports GQA and MLA
4. **Sparse MoE in recurrence only** — Deliberate architectural choice
5. **Agent-driven workflows** — Higher-level orchestration via glasseye
6. **Multi-GPU training from the start** — FSDP baked into reference implementation
---
## Useful Test Commands
```bash
# Run all tests
pytest tests/
# Run specific test class
pytest tests/test_main.py::TestRMSNorm -v
# Run with coverage
pytest tests/ --cov=open_mythos --cov=mythos_glasseye
# Benchmark on CPU/GPU
python tests/small_benchmark.py
```
