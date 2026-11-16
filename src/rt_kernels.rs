//! Kernels accelerating pure-Python hot loops of pyroomacoustics
//! (`simulation/rt.py`). Each kernel performs exactly the same IEEE-754
//! double precision operations, in the same order, as the Python code it
//! replaces, so results are bit-identical.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Recurrence of `poisson_sequence`:
///
/// ```python
/// times = [t0]
/// while times[-1] < t0 + duration:
///     z = rng.uniform()
///     mu = np.minimum(fpcv * (t0 + times[-1]) ** 2, max_rate)
///     dt = np.log(1 / z) / mu
///     times.append(times[-1] + dt)
/// ```
///
/// `logs[i]` must hold `np.log(1 / z_i)`. Returns (times, n_used, done) where
/// `done` is False if `logs` was exhausted before the end condition was met.
#[pyfunction]
fn _poisson_times<'py>(
    py: Python<'py>,
    t0: f64,
    duration: f64,
    fpcv: f64,
    max_rate: f64,
    logs: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, bool)> {
    let logs = logs.as_slice()?;
    let mut times = Vec::with_capacity(logs.len() + 1);
    times.push(t0);
    let mut last = t0;
    let mut used = 0usize;
    let mut done = true;
    while last < t0 + duration {
        if used == logs.len() {
            done = false;
            break;
        }
        let s = t0 + last;
        let a = fpcv * (s * s);
        // np.minimum(a, max_rate)
        let mu = if a.is_nan() {
            a
        } else if a <= max_rate {
            a
        } else {
            max_rate
        };
        let dt = logs[used] / mu;
        used += 1;
        last = last + dt;
        times.push(last);
    }
    Ok((PyArray1::from_vec(py, times), used, done))
}

/// `r_ind = np.sum(z[None, :] > cdf[:, bins], axis=0)` without building the
/// (n_dirs, n_deltas) temporary.
#[pyfunction]
fn _count_greater<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    cdf: PyReadonlyArray2<'py, f64>,
    bins: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let z = z.as_slice()?;
    let bins = bins.as_slice()?;
    let cdf = cdf.as_array();
    let (n_dirs, n_bins) = (cdf.shape()[0], cdf.shape()[1]);
    if z.len() != bins.len() {
        return Err(PyValueError::new_err("z and bins must have the same length"));
    }
    let mut out = Vec::with_capacity(z.len());
    for (j, &b) in bins.iter().enumerate() {
        let b = if b < 0 { b + n_bins as i64 } else { b } as usize;
        let mut cnt = 0i64;
        for d in 0..n_dirs {
            if z[j] > cdf[[d, b]] {
                cnt += 1;
            }
        }
        out.push(cnt);
    }
    Ok(PyArray1::from_vec(py, out))
}

/// The accumulation loop of `directional_sequence`:
///
/// ```python
/// for i, sign, taps in zip(indices, signs, response[r_ind, :]):
///     seq[i : i + taps.shape[0]] += sign * taps * np.sqrt(seq_power_norm[i])
/// ```
#[pyfunction]
fn _seq_accumulate<'py>(
    mut seq: PyReadwriteArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    signs: PyReadonlyArray1<'py, i64>,
    response: PyReadonlyArray2<'py, f64>,
    r_ind: PyReadonlyArray1<'py, i64>,
    seq_power_norm: PyReadonlyArray1<'py, f64>,
) -> PyResult<()> {
    let seq = seq.as_slice_mut()?;
    let indices = indices.as_slice()?;
    let signs = signs.as_slice()?;
    let r_ind = r_ind.as_slice()?;
    let spn = seq_power_norm.as_slice()?;
    let resp = response.as_array();
    let n_taps = resp.shape()[1];
    let n = seq.len();
    for j in 0..indices.len() {
        let i = indices[j];
        if i < 0 || i as usize >= spn.len() {
            return Err(PyValueError::new_err("index out of range"));
        }
        let i = i as usize;
        let s = spn[i].sqrt();
        let sign = signs[j] as f64;
        let r = r_ind[j] as usize;
        for k in 0..n_taps {
            if i + k < n {
                seq[i + k] += (sign * resp[[r, k]]) * s;
            }
        }
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_poisson_times, m)?)?;
    m.add_function(wrap_pyfunction!(_count_greater, m)?)?;
    m.add_function(wrap_pyfunction!(_seq_accumulate, m)?)?;
    Ok(())
}
