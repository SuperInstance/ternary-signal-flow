# ternary-signal-flow

Experiment: ternary signal flow through GPU processing pipeline. Tests quantization, filtering, FFT-like transforms, and reconstruction fidelity with ternary {-1,0,+1} arithmetic.

## Why This Matters

# ternary-signal-flow
Ternary signal flow through a GPU processing pipeline.
Tests quantization, filtering, spectral transforms, and
reconstruction fidelity using only {-1, 0, +1} arithmetic.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub fn quantize
pub fn reconstruct
pub fn sqnr
pub fn ternary_convolve
pub fn ternary_transform
pub enum Stage
pub struct SignalPipeline
pub fn new
pub fn add_stage
pub fn process
pub fn samples_processed
pub fn stage_count
```

## Usage

```toml
[dependencies]
ternary-signal-flow = "0.1.0"
```

```rust
use ternary_signal_flow::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-signal-flow.git
cd ternary-signal-flow
cargo test    # 8 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 204 |
| Public API | 12 items |

## License

Apache-2.0
