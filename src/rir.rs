//! Port of libroom_src/rir_builder.cpp.
//!
//! The original splits the work in `num_threads` contiguous blocks, accumulates
//! each block in its own buffer and then sums the buffers in block order. The
//! floating point result therefore depends on `num_threads`, and we reproduce
//! exactly the same partition and summation order. Unlike the original, we use
//! a persistent thread pool and only allocate/sum the sample range actually
//! touched by each block.

use rayon::prelude::*;

pub trait Real:
    Copy
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    fn pi() -> Self;
    fn from_usize(v: usize) -> Self;
    fn from_i32(v: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn to_i32(self) -> i32;
    /// `(1. - x) * g` where `1.` is a C++ double literal
    fn one_minus_times(x: Self, g: Self) -> Self;
}

impl Real for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const HALF: f32 = 0.5;
    fn pi() -> f32 {
        3.14159265358979323846f32
    }
    fn from_usize(v: usize) -> f32 {
        v as f32
    }
    fn from_i32(v: i32) -> f32 {
        v as f32
    }
    fn sin(self) -> f32 {
        f32::sin(self)
    }
    fn cos(self) -> f32 {
        f32::cos(self)
    }
    fn floor(self) -> f32 {
        f32::floor(self)
    }
    fn ceil(self) -> f32 {
        f32::ceil(self)
    }
    fn to_i32(self) -> i32 {
        self as i32
    }
    fn one_minus_times(x: f32, g: f32) -> f32 {
        ((1.0f64 - x as f64) * g as f64) as f32
    }
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const HALF: f64 = 0.5;
    fn pi() -> f64 {
        3.14159265358979323846f64
    }
    fn from_usize(v: usize) -> f64 {
        v as f64
    }
    fn from_i32(v: i32) -> f64 {
        v as f64
    }
    fn sin(self) -> f64 {
        f64::sin(self)
    }
    fn cos(self) -> f64 {
        f64::cos(self)
    }
    fn floor(self) -> f64 {
        f64::floor(self)
    }
    fn ceil(self) -> f64 {
        f64::ceil(self)
    }
    fn to_i32(self) -> i32 {
        self as i32
    }
    fn one_minus_times(x: f64, g: f64) -> f64 {
        (1.0f64 - x) * g
    }
}

fn sinc_lut<T: Real>(fdl: usize, lut_gran: usize) -> Vec<T> {
    let fdl2 = (fdl - 1) / 2;
    let lut_size = (fdl + 2) * lut_gran;
    let lut_delta = T::ONE / T::from_usize(lut_gran);
    let pi = T::pi();
    let mut lut = vec![T::ZERO; lut_size];
    let mut n = -T::from_usize(fdl2) - T::ONE;
    for v in lut.iter_mut() {
        if n == T::ZERO {
            *v = T::ONE;
        } else {
            *v = (pi * n).sin() / (pi * n);
        }
        n += lut_delta;
    }
    lut
}

fn hann_rir<T: Real>(fdl: usize) -> Vec<T> {
    let pi = T::pi();
    (0..fdl)
        .map(|idx| T::HALF - T::HALF * ((pi * T::from_usize(2 * idx)) / T::from_usize(fdl - 1)).cos())
        .collect()
}

fn hann_fd<T: Real>(fdl: usize) -> Vec<T> {
    let pi = T::pi();
    (0..fdl)
        .map(|idx| T::HALF - T::HALF * ((T::from_usize(2 * idx) * pi) / T::from_usize(fdl - 1)).cos())
        .collect()
}

fn block_size(n: usize, num_threads: usize) -> usize {
    (n as f64 / num_threads as f64).ceil() as usize
}

/// Sum the per-block partial buffers into `out` in block order.
/// `parts[t] = (offset, data)` covers out[offset .. offset + data.len()].
fn ordered_sum<T: Real>(out: &mut [T], parts: &[(usize, Vec<T>)]) {
    const CHUNK: usize = 4096;
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk)| {
        let base = ci * CHUNK;
        // x + 0.0 semantics of the original (turns -0.0 into +0.0)
        for v in chunk.iter_mut() {
            *v = *v + T::ZERO;
        }
        for (off, data) in parts {
            let lo = (*off).max(base);
            let hi = (*off + data.len()).min(base + chunk.len());
            if lo < hi {
                for i in lo..hi {
                    chunk[i - base] += data[i - off];
                }
            }
        }
    });
}

#[derive(Debug)]
pub struct RirError(pub String);

#[allow(clippy::too_many_arguments)]
pub fn rir_builder<T: Real>(
    rir: &mut [T],
    time: &[T],
    alpha: &[T],
    fs: i32,
    fdl: usize,
    lut_gran: usize,
    num_threads: usize,
) -> Result<(), RirError> {
    let n_times = time.len();
    let fdl2 = (fdl.wrapping_sub(1)) / 2;
    let rir_len = rir.len();

    if n_times != alpha.len() {
        return Err(RirError("time and alpha arrays should have the same size".into()));
    }
    if fdl % 2 != 1 {
        return Err(RirError("the fractional filter length should be odd".into()));
    }
    if n_times == 0 {
        // The original dereferences an empty range here (undefined behavior).
        return Ok(());
    }
    let mut t_max = time[0];
    let mut t_min = time[0];
    for &t in &time[1..] {
        if t_max < t {
            t_max = t;
        }
        if t < t_min {
            t_min = t;
        }
    }
    let fsr = T::from_i32(fs);
    let max_sample = ((fsr * t_max).ceil().to_i32() as i64 + fdl2 as i64) as i32;
    let min_sample = ((fsr * t_min).floor().to_i32() as i64 - fdl2 as i64) as i32;
    if min_sample < 0 {
        return Err(RirError("minimum time recorded is less than 0".into()));
    }
    if max_sample >= rir_len as i32 {
        return Err(RirError("the rir array is too small for the maximum time recorded".into()));
    }
    if num_threads == 0 {
        return Ok(());
    }

    let lut = sinc_lut::<T>(fdl, lut_gran);
    let hann = hann_rir::<T>(fdl);
    let lut_gran_f = T::from_usize(lut_gran);
    let bs = block_size(n_times, num_threads);

    let blocks: Vec<(usize, usize)> = (0..num_threads)
        .map(|t| {
            let s = t * bs;
            let e = (s + bs).min(n_times);
            (s, e)
        })
        .filter(|(s, e)| s < e)
        .collect();

    let parts: Vec<(usize, Vec<T>)> = blocks
        .par_iter()
        .map(|&(s, e)| {
            // find range
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            let mut pre: Vec<(i32, i32, T, T)> = Vec::with_capacity(e - s);
            for idx in s..e {
                let sample_frac = fsr * time[idx];
                let time_ip = sample_frac.floor();
                let time_fp = sample_frac - time_ip;
                let x_off_frac = T::one_minus_times(time_fp, lut_gran_f);
                let lut_gran_off = x_off_frac.floor();
                let x_off = x_off_frac - lut_gran_off;
                let lut_pos = lut_gran_off.to_i32();
                let f = (time_ip.to_i32() as i64 - fdl2 as i64) as i32;
                lo = lo.min(f as usize);
                hi = hi.max(f as usize + fdl);
                pre.push((lut_pos, f, x_off, alpha[idx]));
            }
            let mut buf = vec![T::ZERO; hi - lo];
            for (lut_pos, f, x_off, a) in pre {
                let mut lp = lut_pos as usize;
                let base = f as usize - lo;
                for k in 0..fdl {
                    let s0 = lut[lp];
                    let s1 = lut[lp + 1];
                    buf[base + k] += (a * hann[k]) * (s0 + x_off * (s1 - s0));
                    lp += lut_gran;
                }
            }
            (lo, buf)
        })
        .collect();

    ordered_sum(rir, &parts);
    Ok(())
}

pub fn fractional_delay<T: Real>(
    out: &mut [T],
    n_rows: usize,
    n_cols: usize,
    time: &[T],
    lut_gran: usize,
) -> Result<(), RirError> {
    if time.len() != n_rows {
        return Err(RirError("time and output arrays should have the same size".into()));
    }
    let mut fdl = n_cols;
    if fdl % 2 == 0 {
        fdl = fdl.wrapping_sub(1);
    }
    if n_rows == 0 || fdl == 0 || fdl == usize::MAX {
        return Ok(());
    }
    let lut = sinc_lut::<T>(fdl, lut_gran);
    let hann = hann_fd::<T>(fdl);
    let lut_gran_f = T::from_usize(lut_gran);
    out.par_chunks_mut(n_cols).enumerate().for_each(|(idx, row)| {
        let x_off_frac = T::one_minus_times(time[idx], lut_gran_f);
        let lut_gran_off = x_off_frac.floor();
        let x_off = x_off_frac - lut_gran_off;
        let mut lp = lut_gran_off.to_i32() as usize;
        for k in 0..fdl {
            let s0 = lut[lp];
            let s1 = lut[lp + 1];
            row[k] = hann[k] * (s0 + x_off * (s1 - s0));
            lp += lut_gran;
        }
    });
    Ok(())
}

pub fn delay_sum<T: Real>(
    irs: &[T],
    n_irs: usize,
    n_taps: usize,
    delays: &[i32],
    output: &mut [T],
    num_threads: usize,
) -> Result<(), RirError> {
    let out_len = output.len();
    if n_irs != delays.len() {
        return Err(RirError("time and alpha arrays should have the same size".into()));
    }
    if delays.is_empty() {
        return Ok(());
    }
    let d_min = *delays.iter().min().unwrap();
    let d_max = *delays.iter().max().unwrap();
    if d_min < 0 {
        return Err(RirError("delays should be positive".into()));
    }
    if d_max as usize + n_taps > out_len {
        return Err(RirError("the output buffer is too small".into()));
    }
    if num_threads == 0 {
        return Ok(());
    }
    let bs = block_size(n_irs, num_threads);
    let blocks: Vec<(usize, usize)> = (0..num_threads)
        .map(|t| {
            let s = t * bs;
            let e = (s + bs).min(n_irs);
            (s, e)
        })
        .filter(|(s, e)| s < e)
        .collect();
    let parts: Vec<(usize, Vec<T>)> = blocks
        .par_iter()
        .map(|&(s, e)| {
            let lo = delays[s..e].iter().copied().min().unwrap() as usize;
            let hi = delays[s..e].iter().copied().max().unwrap() as usize + n_taps;
            let mut buf = vec![T::ZERO; hi - lo];
            for idx in s..e {
                let d = delays[idx] as usize - lo;
                let row = &irs[idx * n_taps..(idx + 1) * n_taps];
                for k in 0..n_taps {
                    buf[d + k] += row[k];
                }
            }
            (lo, buf)
        })
        .collect();
    ordered_sum(output, &parts);
    Ok(())
}
