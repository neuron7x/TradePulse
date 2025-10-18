use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyErr;
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone)]
pub enum NumericError {
    InvalidWindow,
    InvalidStep,
    EmptyInput { name: &'static str },
    ProbabilityNotFinite(f64),
    ProbabilityOutOfRange(f64),
    UnsupportedMode(String),
}

impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericError::InvalidWindow => write!(f, "window must be greater than zero"),
            NumericError::InvalidStep => write!(f, "step must be greater than zero"),
            NumericError::EmptyInput { name } => write!(f, "{name} must not be empty"),
            NumericError::ProbabilityNotFinite(value) => {
                write!(f, "probability {value} must be finite")
            }
            NumericError::ProbabilityOutOfRange(value) => {
                write!(f, "probability {value} outside [0, 1]")
            }
            NumericError::UnsupportedMode(mode) => {
                write!(f, "unsupported convolution mode '{mode}'")
            }
        }
    }
}

impl std::error::Error for NumericError {}

#[derive(Debug, Clone, Copy)]
pub enum ConvolutionMode {
    Full,
    Same,
    Valid,
}

impl TryFrom<&str> for ConvolutionMode {
    type Error = NumericError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "full" => Ok(Self::Full),
            "same" => Ok(Self::Same),
            "valid" => Ok(Self::Valid),
            other => Err(NumericError::UnsupportedMode(other.to_owned())),
        }
    }
}

fn check_non_empty(array: &[f64], name: &'static str) -> Result<(), NumericError> {
    if array.is_empty() {
        return Err(NumericError::EmptyInput { name });
    }
    Ok(())
}

pub fn sliding_windows_core(
    data: &[f64],
    window: usize,
    step: usize,
) -> Result<(usize, Vec<f64>), NumericError> {
    if window == 0 {
        return Err(NumericError::InvalidWindow);
    }
    if step == 0 {
        return Err(NumericError::InvalidStep);
    }
    if data.len() < window {
        return Ok((0, Vec::new()));
    }
    let windows = ((data.len() - window) / step) + 1;
    let mut values = Vec::with_capacity(windows * window);
    for i in 0..windows {
        let start = i * step;
        values.extend_from_slice(&data[start..start + window]);
    }
    Ok((windows, values))
}

pub fn quantiles_core(data: &[f64], probabilities: &[f64]) -> Result<Vec<f64>, NumericError> {
    if let Some(&value) = probabilities.iter().find(|p| !p.is_finite()) {
        return Err(NumericError::ProbabilityNotFinite(value));
    }
    if let Some(&value) = probabilities.iter().find(|&&p| p < 0.0 || p > 1.0) {
        return Err(NumericError::ProbabilityOutOfRange(value));
    }
    if data.is_empty() {
        return Ok(vec![f64::NAN; probabilities.len()]);
    }
    let mut values = data.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = values.len();
    let mut results = Vec::with_capacity(probabilities.len());
    for &probability in probabilities {
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
    Ok(results)
}

pub fn full_convolution(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
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

pub fn convolve_core(
    signal: &[f64],
    kernel: &[f64],
    mode: ConvolutionMode,
) -> Result<Vec<f64>, NumericError> {
    check_non_empty(signal, "signal")?;
    check_non_empty(kernel, "kernel")?;
    let full = full_convolution(signal, kernel);
    let (n, m) = (signal.len(), kernel.len());
    let full_len = full.len();
    let result = match mode {
        ConvolutionMode::Full => full,
        ConvolutionMode::Same => {
            let target = n.max(m);
            let pad = (full_len - target) / 2;
            let start = pad;
            let end = start + target;
            full[start..end].to_vec()
        }
        ConvolutionMode::Valid => {
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
    };
    Ok(result)
}

impl From<NumericError> for PyErr {
    fn from(value: NumericError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

#[pyfunction]
fn sliding_windows<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window: usize,
    step: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let slice = data.as_slice()?;
    let (rows, values) = sliding_windows_core(slice, window, step)?;
    let array = Array2::from_shape_vec((rows, window), values)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array_bound(py, array))
}

#[pyfunction]
fn quantiles<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    probabilities: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let result = quantiles_core(slice, &probabilities)?;
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
fn convolve<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    kernel: PyReadonlyArray1<'py, f64>,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal_slice = signal.as_slice()?;
    let kernel_slice = kernel.as_slice()?;
    let mode = ConvolutionMode::try_from(mode)?;
    let result = convolve_core(signal_slice, kernel_slice, mode)?;
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pymodule]
fn tradepulse_accel(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sliding_windows, module)?)?;
    module.add_function(wrap_pyfunction!(quantiles, module)?)?;
    module.add_function(wrap_pyfunction!(convolve, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add(
        "__doc__",
        "Rust accelerators for TradePulse numeric primitives",
    )?;
    module.add("PYTHON_IMPLEMENTATION", "rust")?;
    module.add("PYTHON_VERSION", py.version())?;
    Ok(())
}
