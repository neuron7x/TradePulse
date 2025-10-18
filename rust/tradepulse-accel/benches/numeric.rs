use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tradepulse_accel::{convolve_core, quantiles_core, sliding_windows_core, ConvolutionMode};

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_millis(600))
        .warm_up_time(Duration::from_millis(200))
        .noise_threshold(0.01)
        .confidence_level(0.99)
        .significance_level(0.05)
        .sample_size(40)
        .configure_from_args()
}

fn bench_sliding_windows(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_windows");
    let sizes = [1_024usize, 16_384, 131_072];
    let window = 128usize;
    let step = 4usize;
    for &size in &sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let (_rows, result) =
                    sliding_windows_core(black_box(data.as_slice()), window, step)
                        .expect("valid window parameters");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantiles");
    let sizes = [1_024usize, 65_536, 262_144];
    let probabilities = [0.1, 0.5, 0.9, 0.99];
    for &size in &sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.125).cos()).collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let result = quantiles_core(black_box(data.as_slice()), &probabilities)
                    .expect("valid quantile configuration");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_convolve_same(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve_same");
    let configs = [(2_048usize, 64usize), (16_384, 128), (131_072, 256)];
    for &(signal_len, kernel_len) in &configs {
        let signal: Vec<f64> = (0..signal_len).map(|i| (i as f64 * 0.01).sin()).collect();
        let kernel: Vec<f64> = (0..kernel_len).map(|i| (i as f64 * 0.02).cos()).collect();
        let data = (signal, kernel);
        let bench_id = BenchmarkId::new(format!("n{signal_len}"), kernel_len);
        group.throughput(Throughput::Elements((signal_len + kernel_len) as u64));
        group.bench_with_input(bench_id, &data, |b, (signal, kernel)| {
            b.iter(|| {
                let result = convolve_core(
                    black_box(signal.as_slice()),
                    black_box(kernel.as_slice()),
                    ConvolutionMode::Same,
                )
                .expect("valid convolution inputs");
                black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = numeric_benches;
    config = configure_criterion();
    targets = bench_sliding_windows, bench_quantiles, bench_convolve_same
);
criterion_main!(numeric_benches);
