//! Fast (non bit-exact) fused kernels.
//!
//! `ism_rir_bands` replaces the numpy code of `simulation.ism.compute_ism_rir`
//! for omnidirectional sources/receivers plus the per-band calls to
//! `rir_builder`: distances, delays, damping, air absorption and the
//! windowed-sinc fractional delays are computed in a single parallel pass.
//! Results equal the original up to floating point rounding (summation order).

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn sinc_lut(fdl: usize, lut_gran: usize) -> Vec<f32> {
    let fdl2 = (fdl - 1) / 2;
    let lut_size = (fdl + 2) * lut_gran;
    let delta = 1.0f32 / lut_gran as f32;
    let pi = std::f32::consts::PI;
    let mut n = -(fdl2 as f32) - 1.0;
    let mut lut = vec![0.0f32; lut_size];
    for v in lut.iter_mut() {
        *v = if n == 0.0 { 1.0 } else { (pi * n).sin() / (pi * n) };
        n += delta;
    }
    lut
}

fn hann(fdl: usize) -> Vec<f32> {
    let pi = std::f32::consts::PI;
    (0..fdl)
        .map(|i| 0.5 - 0.5 * ((pi * (2 * i) as f32) / (fdl - 1) as f32).cos())
        .collect()
}

/// Returns an array (n_bands, N) with the per-band ISM impulse responses,
/// where `N = ceil(t_max * fs + fdl2 + 1) + 1` exactly like `compute_ism_rir`.
#[pyfunction]
#[pyo3(signature = (images, damping, visible, mic, c, fs, fdl, lut_gran, air_abs=None))]
#[allow(clippy::too_many_arguments)]
fn ism_rir_bands<'py>(
    py: Python<'py>,
    images: PyReadonlyArray2<'py, f64>,
    damping: PyReadonlyArray2<'py, f64>,
    visible: PyReadonlyArray1<'py, bool>,
    mic: PyReadonlyArray1<'py, f64>,
    c: f64,
    fs: f64,
    fdl: usize,
    lut_gran: usize,
    air_abs: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let images = images.as_array();
    let damping = damping.as_array();
    let visible = visible.as_slice()?;
    let mic = mic.as_slice()?.to_vec();
    let air: Option<Vec<f64>> = air_abs.map(|a| a.as_slice().map(|s| s.to_vec())).transpose()?;
    let dim = images.shape()[0];
    let n_img = images.shape()[1];
    let nb = damping.shape()[0];
    if damping.shape()[1] != n_img || visible.len() != n_img || mic.len() != dim {
        return Err(PyValueError::new_err("inconsistent array shapes"));
    }
    if fdl % 2 == 0 {
        return Err(PyValueError::new_err("the fractional filter length should be odd"));
    }
    if let Some(a) = &air {
        if a.len() != nb {
            return Err(PyValueError::new_err("air absorption must have one coefficient per band"));
        }
    }
    let fdl2 = fdl / 2;
    let delay = fdl2 as f64 / fs;

    // gather visible images: (time, amplitudes)
    let idx: Vec<usize> = (0..n_img).filter(|&i| visible[i]).collect();
    let n = idx.len();
    let mut times = vec![0.0f64; n];
    let mut amps = vec![0.0f32; n * nb];
    crate::pool::install(|| times.par_iter_mut().zip(amps.par_chunks_mut(nb.max(1))).enumerate().for_each(|(j, (t, a))| {
        let i = idx[j];
        let mut d2 = 0.0f64;
        for k in 0..dim {
            let d = images[[k, i]] - mic[k];
            d2 += d * d;
        }
        let dist = d2.sqrt();
        *t = dist / c + delay;
        for b in 0..nb {
            let mut v = damping[[b, i]] / dist;
            if let Some(air) = &air {
                v *= (-0.5 * air[b] * dist).exp();
            }
            a[b] = v as f32;
        }
    }));
    let t_max = times.iter().cloned().fold(0.0f64, f64::max);
    let big_n = if n == 0 {
        (fdl2 as f64 + 1.0).ceil() as usize + 1
    } else {
        (t_max * fs + fdl2 as f64 + 1.0).ceil() as usize + 1
    };

    let lut = sinc_lut(fdl, lut_gran);
    let win = hann(fdl);
    // Window-premultiplied, transposed tables: for a LUT offset g,
    // w[k] = tw[g][k] + x_off * td[g][k] with contiguous k (vectorizable).
    let mut tw = vec![0.0f32; (lut_gran + 1) * fdl];
    let mut td = vec![0.0f32; (lut_gran + 1) * fdl];
    for g in 0..=lut_gran {
        for k in 0..fdl {
            let a = lut[g + k * lut_gran];
            let b = lut[g + 1 + k * lut_gran];
            tw[g * fdl + k] = win[k] * a;
            td[g * fdl + k] = win[k] * (b - a);
        }
    }
    let n_threads = crate::pool::num_threads();
    // number of partial buffers: enough parallelism without huge memory
    let n_parts = n_threads.min((n / 1024).max(1));
    let part = n.div_ceil(n_parts.max(1)).max(1);

    let out: Vec<f32> = py.detach(|| crate::pool::install(|| {
        // interleaved partial buffers (N x nb)
        let partials: Vec<Vec<f32>> = (0..n_parts)
            .into_par_iter()
            .map(|p| {
                let mut buf = vec![0.0f32; big_n * nb];
                let mut w = vec![0.0f32; fdl];
                let s = p * part;
                let e = ((p + 1) * part).min(n);
                for j in s..e {
                    let sample_frac = times[j] * fs;
                    let ip = sample_frac.floor();
                    let fp = (sample_frac - ip) as f32;
                    let x_off_frac = (1.0 - fp) * lut_gran as f32;
                    let off = x_off_frac.floor();
                    let x_off = x_off_frac - off;
                    let g = (off as usize).min(lut_gran);
                    let (twg, tdg) = (&tw[g * fdl..(g + 1) * fdl], &td[g * fdl..(g + 1) * fdl]);
                    for k in 0..fdl {
                        w[k] = twg[k] + x_off * tdg[k];
                    }
                    let start = ip as isize - fdl2 as isize;
                    let a = &amps[j * nb..(j + 1) * nb];
                    // clip the filter to the buffer
                    let k0 = (-start).max(0) as usize;
                    let k1 = ((big_n as isize - start).min(fdl as isize)).max(0) as usize;
                    if k0 >= k1 {
                        continue;
                    }
                    let base = (start + k0 as isize) as usize;
                    if nb == 1 {
                        let a0 = a[0];
                        let dst = &mut buf[base..base + (k1 - k0)];
                        for (d, wk) in dst.iter_mut().zip(&w[k0..k1]) {
                            *d += a0 * wk;
                        }
                    } else {
                        for k in k0..k1 {
                            let row = &mut buf[(base + k - k0) * nb..(base + k - k0 + 1) * nb];
                            let wk = w[k];
                            for b in 0..nb {
                                row[b] += a[b] * wk;
                            }
                        }
                    }
                }
                buf
            })
            .collect();
        // reduce + transpose to (nb, N)
        let mut out = vec![0.0f32; nb * big_n];
        const CH: usize = 2048;
        let chunks: Vec<(usize, Vec<f32>)> = (0..big_n.div_ceil(CH))
            .into_par_iter()
            .map(|ci| {
                let t0 = ci * CH;
                let t1 = (t0 + CH).min(big_n);
                let mut loc = vec![0.0f32; (t1 - t0) * nb];
                for pb in &partials {
                    for (l, v) in loc.iter_mut().zip(&pb[t0 * nb..t1 * nb]) {
                        *l += *v;
                    }
                }
                (t0, loc)
            })
            .collect();
        for (t0, loc) in chunks {
            let len = loc.len() / nb.max(1);
            for t in 0..len {
                for b in 0..nb {
                    out[b * big_n + t0 + t] = loc[t * nb + b];
                }
            }
        }
        out
    }));
    let arr = PyArray2::<f32>::zeros(py, [nb, big_n], false);
    unsafe { arr.as_slice_mut()? }.copy_from_slice(&out);
    Ok(arr)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ism_rir_bands, m)?)?;
    Ok(())
}
