//! # ternary-signal-flow
//!
//! Ternary signal flow through a GPU processing pipeline.
//! Tests quantization, filtering, spectral transforms, and
//! reconstruction fidelity using only {-1, 0, +1} arithmetic.

use std::collections::VecDeque;

/// Quantize a real-valued signal to ternary.
pub fn quantize(signal: &[f64]) -> Vec<i8> {
    signal.iter().map(|&x| {
        if x > 0.33 { 1 } else if x < -0.33 { -1 } else { 0 }
    }).collect()
}

/// Reconstruct approximate signal from ternary.
pub fn reconstruct(ternary: &[i8]) -> Vec<f64> {
    ternary.iter().map(|&t| t as f64).collect()
}

/// Signal-to-quantization-noise ratio in dB.
pub fn sqnr(original: &[f64], reconstructed: &[f64]) -> f64 {
    let signal_power: f64 = original.iter().map(|&x| x * x).sum();
    let noise_power: f64 = original.iter().zip(reconstructed)
        .map(|(&o, &r)| (o - r) * (o - r)).sum();
    if noise_power == 0.0 { return f64::INFINITY; }
    10.0 * (signal_power / noise_power).log10()
}

/// Ternary filter: convolve with ternary kernel.
pub fn ternary_convolve(signal: &[i8], kernel: &[i8]) -> Vec<i8> {
    let mut output = Vec::with_capacity(signal.len());
    for i in 0..signal.len() {
        let mut acc = 0i8;
        for (j, &k) in kernel.iter().enumerate() {
            if i + j < signal.len() {
                let prod = signal[i + j] * k; // standard multiply
                acc = acc.saturating_add(prod);
            }
        }
        // Re-ternarize: clamp to {-1, 0, +1}
        output.push(if acc > 0 { 1 } else if acc < 0 { -1 } else { 0 });
    }
    output
}

/// Ternary Hadamard-like transform (simplified).
/// Computes pairwise ternary products in a butterfly pattern.
pub fn ternary_transform(signal: &[i8]) -> Vec<i8> {
    let n = signal.len();
    if n <= 1 { return signal.to_vec(); }
    let mut result = signal.to_vec();
    let mut stride = 1;
    while stride < n {
        for i in (0..n).step_by(2 * stride) {
            if i + stride < n {
                let a = result[i];
                let b = result[i + stride];
                result[i] = tadd(a, b);
                result[i + stride] = tsub(a, b);
            }
        }
        stride *= 2;
    }
    result
}

fn tadd(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1, (-1, 0) => -1, (-1, 1) => 0,
        (0, -1) => -1, (0, 0) => 0, (0, 1) => 1,
        (1, -1) => 0, (1, 0) => 1, (1, 1) => -1,
        _ => 0,
    }
}

fn tsub(a: i8, b: i8) -> i8 { tadd(a, -b) }

/// A processing pipeline stage.
#[derive(Debug, Clone)]
pub enum Stage {
    Quantize { threshold: f64 },
    Filter { kernel: Vec<i8> },
    Transform,
    Delay { samples: usize },
    Threshold { min: i8 },
}

/// Signal pipeline that processes ternary data.
pub struct SignalPipeline {
    stages: Vec<Stage>,
    buffer: VecDeque<i8>,
    samples_processed: u64,
}

impl SignalPipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new(), buffer: VecDeque::new(), samples_processed: 0 }
    }

    pub fn add_stage(&mut self, stage: Stage) { self.stages.push(stage); }

    pub fn process(&mut self, input: &[i8]) -> Vec<i8> {
        let mut signal = input.to_vec();
        for stage in &self.stages {
            signal = match stage {
                Stage::Filter { kernel } => ternary_convolve(&signal, kernel),
                Stage::Transform => ternary_transform(&signal),
                Stage::Threshold { min } => signal.iter().map(|&v| if v >= *min { v } else { 0 }).collect(),
                Stage::Delay { samples } => {
                    let mut delayed = vec![0i8; *samples];
                    delayed.extend(&signal);
                    for &s in signal.iter().take(*samples) { self.buffer.push_back(s); }
                    delayed
                }
                Stage::Quantize { .. } => signal, // already ternary
            };
        }
        self.samples_processed += input.len() as u64;
        signal
    }

    pub fn samples_processed(&self) -> u64 { self.samples_processed }
    pub fn stage_count(&self) -> usize { self.stages.len() }
}

impl Default for SignalPipeline {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize() {
        let signal = vec![0.5, -0.7, 0.1, -0.2, 0.9];
        let t = quantize(&signal);
        assert_eq!(t, vec![1, -1, 0, 0, 1]);
    }

    #[test]
    fn test_round_trip() {
        let original = vec![0.5, -0.5, 0.0, 1.0, -1.0];
        let t = quantize(&original);
        let r = reconstruct(&t);
        assert_eq!(r.len(), 5);
        assert!(sqnr(&original, &r) > 0.0);
    }

    #[test]
    fn test_convolve_identity() {
        let signal = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let kernel = vec![1]; // identity kernel
        let result = ternary_convolve(&signal, &kernel);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_convolve_smoothing() {
        let signal = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let kernel = vec![1, 1]; // low-pass
        let result = ternary_convolve(&signal, &kernel);
        // High-frequency oscillation should be reduced
        assert!(result.iter().filter(|&&v| v == 0).count() > 0);
    }

    #[test]
    fn test_transform_power_of_two() {
        let signal = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let transformed = ternary_transform(&signal);
        assert_eq!(transformed.len(), 8);
        // All values should be valid ternary
        for &v in &transformed { assert!(v >= -1 && v <= 1); }
    }

    #[test]
    fn test_pipeline_multi_stage() {
        let mut pipeline = SignalPipeline::new();
        pipeline.add_stage(Stage::Filter { kernel: vec![1, 0, -1] });
        pipeline.add_stage(Stage::Threshold { min: 1 });
        let input = vec![1, 1, 1, 0, -1, -1, 0, 1];
        let output = pipeline.process(&input);
        assert_eq!(pipeline.stage_count(), 2);
        assert_eq!(pipeline.samples_processed(), 8);
    }

    #[test]
    fn test_sqnr_perfect() {
        let signal = vec![1.0, -1.0, 0.0];
        let t = quantize(&signal);
        let r = reconstruct(&t);
        assert!(sqnr(&signal, &r).is_infinite()); // perfect reconstruction
    }

    #[test]
    fn test_pipeline_empty() {
        let mut pipeline = SignalPipeline::new();
        let input = vec![1, -1, 0];
        let output = pipeline.process(&input);
        assert_eq!(output, input); // no stages = passthrough
    }
}
