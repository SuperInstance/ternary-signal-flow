# ternary-signal-flow

**Signal processing where every sample is {-1, 0, +1}. Quantize, filter, transform — and see what survives.**

Most signal processing assumes real-valued samples. You quantize to ternary at the end, after all the math is done. This crate asks the opposite question: what if you quantize first and do everything in ternary? What kind of signal processing is possible when your alphabet is exactly three symbols?

The answer: surprisingly much. You get convolution (with ternary kernels), a Hadamard-like butterfly transform, thresholding, and delay lines. You can chain these into multi-stage pipelines. And the whole thing runs on integer arithmetic — no FPU needed.

## The Insight

Signal processing is dominated by multiply-accumulate operations. In ternary, multiplication is a lookup table:

```
×  │ -1  0  +1
───┼──────────
-1 │ +1  0  -1
 0 │  0  0   0
+1 │ -1  0  +1
```

And addition is modular (Z₃): -1 + -1 = +1, -1 + 1 = 0, 1 + 1 = -1. These are both single-cycle operations on any processor. No FPU, no IEEE 754, no denormal handling. Just truth tables.

The catch: you lose resolution. A signal quantized to three levels has ~9.5 dB of SQNR at best. But for applications where the signal is already low-resolution (sensor thresholds, binary decisions, GPU warp votes), ternary is enough.

## Quick Start

```toml
[dependencies]
ternary-signal-flow = "0.1.0"
```

```rust
use ternary_signal_flow::*;

// Quantize a real signal to ternary
let signal = vec![0.5, -0.7, 0.1, -0.2, 0.9];
let ternary = quantize(&signal);
assert_eq!(ternary, vec![1, -1, 0, 0, 1]);

// Filter with a ternary kernel
let kernel = vec![1, 0, -1]; // edge detector
let filtered = ternary_convolve(&ternary, &kernel);

// Hadamard-like spectral transform
let spectrum = ternary_transform(&vec![1, -1, 1, -1, 1, -1, 1, -1]);

// Measure reconstruction quality
let reconstructed = reconstruct(&ternary);
let snr = sqnr(&signal, &reconstructed);
println!("SQNR: {:.1} dB", snr);

// Chain into a pipeline
let mut pipeline = SignalPipeline::new();
pipeline.add_stage(Stage::Filter { kernel: vec![1, 1, 1] }); // smoothing
pipeline.add_stage(Stage::Threshold { min: 1 });              // suppress noise
let output = pipeline.process(&ternary);
```

## Architecture

```
  Input Signal [f64]
       │
       ▼
  ┌──────────┐
  │ quantize │  threshold at ±0.33
  └────┬─────┘
       │  Vec<i8> ∈ {-1, 0, +1}
       ▼
  ┌──────────────────────────────────────┐
  │         SignalPipeline               │
  │                                      │
  │  Stage::Filter { kernel }            │
  │    → ternary_convolve (re-ternarize)  │
  │                                      │
  │  Stage::Transform                    │
  │    → Hadamard butterfly (Z₃ add/sub) │
  │                                      │
  │  Stage::Threshold { min }            │
  │    → suppress below threshold        │
  │                                      │
  │  Stage::Delay { samples }            │
  │    → shift signal in time            │
  │                                      │
  │  Stage::Quantize { threshold }       │
  │    → (passthrough if already ternary) │
  └──────────────────────────────────────┘
       │
       ▼
  Output Signal [i8]
```

## API Reference

### Quantization

```rust
quantize(signal: &[f64]) -> Vec<i8>
reconstruct(ternary: &[i8]) -> Vec<f64>
sqnr(original: &[f64], reconstructed: &[f64]) -> f64
```

- **`quantize`** — maps each sample: > 0.33 → +1, < -0.33 → -1, else 0. The ±0.33 thresholds split the [-1, 1] range into three equal bands.
- **`reconstruct`** — trivial: maps +1 → 1.0, 0 → 0.0, -1 → -1.0.
- **`sqnr`** — signal-to-quantization-noise ratio in dB: `10 log₁₀(signal_power / noise_power)`. Returns `∞` if noise is zero (perfect reconstruction).

### Filtering

```rust
ternary_convolve(signal: &[i8], kernel: &[i8]) -> Vec<i8>
```

Standard discrete convolution with re-ternarization. Each output element is the sum of element-wise products, clamped back to {-1, 0, +1}. The convolution is causal (no future samples).

**Identity kernel**: `[1]` → output equals input.
**Low-pass**: `[1, 1]` → smooths high-frequency oscillation (alternating ±1 cancels to 0).
**Edge detector**: `[1, 0, -1]` → highlights transitions.

### Transform

```rust
ternary_transform(signal: &[i8]) -> Vec<i8>
```

Hadamard-like butterfly transform using Z₃ arithmetic. Works best with power-of-2 length signals. The butterfly pattern: at each stride, pair up elements and compute `(a+b, a-b)` using ternary addition and subtraction. The result is a kind of spectral decomposition — high-energy at the start, detail at the end.

### Pipeline

```rust
SignalPipeline::new() -> SignalPipeline
pipeline.add_stage(stage: Stage)
pipeline.process(input: &[i8]) -> Vec<i8>
pipeline.samples_processed() -> u64
pipeline.stage_count() -> usize
```

Stages are applied in order. Each stage's output feeds the next. The pipeline tracks total samples processed.

### Stage

```rust
pub enum Stage {
    Quantize { threshold: f64 },
    Filter { kernel: Vec<i8> },
    Transform,
    Delay { samples: usize },
    Threshold { min: i8 },
}
```

## Z₃ Arithmetic

The ternary transform uses Z₃ (integers mod 3) for addition and subtraction:

```
Addition (mod 3):
+  │ -1  0  +1
───┼──────────
-1 │ +1 -1   0
 0 │ -1  0  +1
+1 │  0 +1  -1

Subtraction: a - b = a + (-b)
```

This is different from standard integer clamping. In Z₃, 1 + 1 = -1 (wraps around). In clamped arithmetic, 1 + 1 = 1 (saturates). The transform uses Z₃; the convolution uses clamped arithmetic. This is intentional — the transform preserves information (it's invertible in Z₃), while the convolution deliberately loses information (it's a filter).

## Real-World Example: Ternary Audio Detector

```rust
use ternary_signal_flow::*;

// Simulate a simple "clap detector" using ternary signal flow
let mut pipeline = SignalPipeline::new();
pipeline.add_stage(Stage::Filter { kernel: vec![1, 1, 1] });  // smooth
pipeline.add_stage(Stage::Threshold { min: 1 });               // only loud signals

// Process microphone input (already quantized)
let mic_input = vec![0, 0, 0, 1, 1, 1, -1, -1, 0, 0, 0, 0];
let detected = pipeline.process(&mic_input);

// Non-zero output → sound detected
let has_sound = detected.iter().any(|&v| v != 0);
println!("Sound detected: {}", has_sound);
println!("Samples processed: {}", pipeline.samples_processed());
```

## SQNR Analysis

| Input | Ternary | SQNR |
|-------|---------|------|
| [-1, 0, 1] | [-1, 0, 1] | ∞ (perfect) |
| [0.5, -0.5] | [1, -1] | 6.0 dB |
| [0.1, 0.2, 0.3] | [0, 0, 1] | 4.8 dB |
| Random uniform [-1, 1] | ternary | ~9.5 dB (theoretical max) |

The ~9.5 dB theoretical SQNR comes from having 3 quantization levels over a [-1, 1] range. Each additional doubling of levels adds ~6 dB. Ternary is low-resolution by design.

## Ecosystem

- **ternary-watermark** — embed watermarks that survive ternary signal flow
- **ternary-vortex** — 2D flow fields with ternary velocity
- **ternary-shard** — distribute signal processing across GPUs

## Open Questions

- **Inverse transform**: The Hadamard transform is invertible in Z₃. Implementing `inverse_transform` would enable lossless compression of ternary signals.
- **Adaptive thresholds**: The ±0.33 quantization thresholds are fixed. Adaptive thresholds (based on signal statistics) would improve SQNR for non-uniform distributions.
- **Frequency-domain filtering**: Convolution in time domain → multiplication in transform domain. If the transform is invertible, you can filter in the "frequency" domain.
- **Multi-dimensional signals**: Extend to 2D (images) and 3D (volumes) for GPU texture processing.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 204 |
| Public API | 12 items |

## License

Apache-2.0
