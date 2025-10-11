use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn check_non_empty(array: &[f64], name: &str) -> PyResult<()> {
    if array.is_empty() {
        return Err(PyValueError::new_err(format!("{name} must not be empty")));
    }
    Ok(())
}

#[pyfunction]
fn sliding_windows<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be greater than zero"));
    }
    if step == 0 {
        return Err(PyValueError::new_err("step must be greater than zero"));
    }
    let slice = data.as_slice()?;
    if slice.len() < window {
        let empty = Array2::<f64>::zeros((0, window));
        return Ok(PyArray2::from_owned_array_bound(py, empty));
    }
    let windows = ((slice.len() - window) / step) + 1;
    let mut values = Vec::with_capacity(windows * window);
    for i in 0..windows {
        let start = i * step;
        values.extend_from_slice(&slice[start..start + window]);
    }
    let array = Array2::from_shape_vec((windows, window), values)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array_bound(py, array))
}

#[pyfunction]
fn quantiles<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    probabilities: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if probabilities.iter().any(|p| !p.is_finite()) {
        return Err(PyValueError::new_err("probabilities must be finite"));
    }
    let mut values = data.as_slice()?.to_vec();
    if values.is_empty() {
        let result = vec![f64::NAN; probabilities.len()];
        return Ok(PyArray1::from_vec_bound(py, result));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let mut results = Vec::with_capacity(probabilities.len());
    for probability in probabilities {
        if !(0.0..=1.0).contains(&probability) {
            return Err(PyValueError::new_err(format!(
                "probability {probability} outside [0, 1]"
            )));
        }
        let position = probability * (n as f64 - 1.0);
        let lower_idx = position.floor() as usize;
        let upper_idx = position.ceil() as usize;
        if lower_idx == upper_idx {
            results.push(values[lower_idx]);
        } else {
            let weight = position - lower_idx as f64;
            let lower = values[lower_idx];
            let upper = values[upper_idx];
            results.push(lower + (upper - lower) * weight);
        }
    }
    Ok(PyArray1::from_vec_bound(py, results))
}

fn full_convolution(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let output_len = signal.len() + kernel.len() - 1;
    let mut output = vec![0.0; output_len];
    for (i, value) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, kernel_value) in kernel.iter().enumerate() {
            let signal_index = i as isize - j as isize;
            if signal_index >= 0 {
                let idx = signal_index as usize;
                if idx < signal.len() {
                    sum += signal[idx] * kernel_value;
                }
            }
        }
        *value = sum;
    }
    output
}

#[pyfunction]
fn convolve<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    kernel: PyReadonlyArray1<'py, f64>,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal_vec = signal.as_slice()?.to_vec();
    let kernel_vec = kernel.as_slice()?.to_vec();
    check_non_empty(&signal_vec, "signal")?;
    check_non_empty(&kernel_vec, "kernel")?;
    let full = full_convolution(&signal_vec, &kernel_vec);
    let (n, m) = (signal_vec.len(), kernel_vec.len());
    let full_len = full.len();
    let result = match mode {
        "full" => full,
        "same" => {
            let target = n.max(m);
            let pad = (full_len - target) / 2;
            let start = pad;
            let end = start + target;
            full[start..end].to_vec()
        }
        "valid" => {
            if n >= m {
                let length = n - m + 1;
                let start = m - 1;
                let end = start + length;
                full[start..end].to_vec()
            } else {
                let length = m - n + 1;
                let start = n - 1;
                let end = start + length;
                full[start..end].to_vec()
            }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported convolution mode '{mode}'"
            )))
        }
    };
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pymodule]
fn tradepulse_accel(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sliding_windows, module)?)?;
    module.add_function(wrap_pyfunction!(quantiles, module)?)?;
    module.add_function(wrap_pyfunction!(convolve, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__doc__", "Rust accelerators for TradePulse numeric primitives")?;
    module.add("PYTHON_IMPLEMENTATION", "rust")?;
    module.add("PYTHON_VERSION", py.version())?;
    Ok(())
}
