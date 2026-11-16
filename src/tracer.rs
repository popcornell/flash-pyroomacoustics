//! Batch ray tracing driver.
//!
//! Bit-exactness constraints:
//! * The histograms are sums of floats: entries must be accumulated in the
//!   same order as the original serial loop (ray by ray, bounce by bounce,
//!   microphone by microphone).
//! * When walls scatter, every bounce draws from the global Mersenne twister,
//!   so rays must consume the RNG stream in order.
//!
//! Rays are traced (possibly in parallel) into event logs that are applied to
//! the histograms in the original order.

use crate::mic::{Histogram2D, MicData};
use crate::room::{EventLog, Geometry, Params, Paths};
use crate::rng;
use crate::wall::{ENERGY_0, HALF_PI, PI, TWO_PI};
use rayon::prelude::*;

/// Opt-in: give every ray its own random generator (not bit-identical to the
/// original, but allows parallel ray tracing with scattering).
pub static PARALLEL_RNG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub enum Energies {
    Uniform(Vec<f32>),
    PerRay(Vec<Vec<f32>>),
}

impl Energies {
    #[inline]
    fn get(&self, i: usize) -> &[f32] {
        match self {
            Energies::Uniform(v) => v,
            Energies::PerRay(v) => &v[i],
        }
    }
}

#[inline]
pub fn angles_to_dir<const D: usize>(phi: f32, theta: f32) -> [f32; D] {
    let mut d = [0.0f32; D];
    if D == 2 {
        d[0] = phi.cos();
        d[1] = phi.sin();
    } else {
        d[0] = theta.sin() * phi.cos();
        d[1] = theta.sin() * phi.sin();
        d[2] = theta.cos();
    }
    d
}

/// Directions used by `ray_tracing(n_rays, source)`.
pub fn fibonacci_dirs<const D: usize>(n_rays: usize) -> (Vec<[f32; D]>, f32) {
    let energy_value = ENERGY_0 / (n_rays as f32);
    let mut dirs = Vec::with_capacity(n_rays);
    if D == 3 {
        let offset = 2.0f32 / (n_rays as f32);
        let increment = PI * (3.0f32 - 5.0f32.sqrt());
        for i in 0..n_rays {
            let z = ((i as f32) * offset - 1.0) + offset / 2.0f32;
            let rho = (1.0f32 - z * z).sqrt();
            let phi = (i as f32) * increment;
            let x = phi.cos() * rho;
            let y = phi.sin() * rho;
            let azimuth = y.atan2(x);
            let colatitude = (x * x + y * y).sqrt().atan2(z);
            dirs.push(angles_to_dir::<D>(azimuth, colatitude));
        }
    } else {
        let offset = TWO_PI / (n_rays as f32);
        for i in 0..n_rays {
            dirs.push(angles_to_dir::<D>((i as f32) * offset, 0.0));
        }
    }
    (dirs, energy_value)
}

/// Directions used by `ray_tracing(nb_phis, nb_thetas, source)`.
pub fn grid_dirs<const D: usize>(nb_phis: usize, nb_thetas: usize) -> (Vec<[f32; D]>, f32) {
    let energy_value = ENERGY_0 / ((nb_phis * nb_thetas) as f32);
    let mut dirs = vec![];
    for i in 0..nb_phis {
        let phi = TWO_PI * (i as f32) / (nb_phis as f32);
        for j in 0..nb_thetas {
            let mut theta = (2.0f32 * ((j as f32) / (nb_thetas as f32)) - 1.0f32).acos();
            if D == 2 {
                theta = HALF_PI;
            }
            dirs.push(angles_to_dir::<D>(phi, theta));
            if D == 2 {
                break;
            }
        }
    }
    (dirs, energy_value)
}

fn apply(log: &EventLog, hists: &mut [&mut Vec<Histogram2D>]) {
    if log.events.is_empty() {
        return;
    }
    let nb = log.energies.len() / log.events.len();
    for (i, e) in log.events.iter().enumerate() {
        hists[e.mic as usize][e.dir as usize].log_col(e.col as usize, &log.energies[i * nb..(i + 1) * nb]);
    }
}

pub fn trace_rays<const D: usize>(
    geo: &Geometry<D>,
    params: &Params,
    mics: &[MicData<D>],
    hists: &mut [&mut Vec<Histogram2D>],
    dirs: &[[f32; D]],
    energies: &Energies,
    source: &[f32; D],
) {
    let n = dirs.len();
    if n == 0 {
        return;
    }
    let parallel_rng = PARALLEL_RNG.load(std::sync::atomic::Ordering::Relaxed);
    if !geo.any_wall_scatters() || parallel_rng {
        // Without scattering the RNG is never used and the result is exact.
        // With the (opt-in) parallel RNG mode, ray i uses its own generator
        // seeded from one draw of the global engine: statistically equivalent
        // to the original but not bit-identical.
        let base_seed = if geo.any_wall_scatters() { rng::with_engine(|r| r.next_u64()) } else { 0 };
        // No randomness involved: rays are independent.
        const CHUNK: usize = 64;
        const SUPER: usize = CHUNK * 1024;
        let mut start = 0;
        while start < n {
            let end = (start + SUPER).min(n);
            let logs: Vec<EventLog> = (start..end)
                .into_par_iter()
                .step_by(CHUNK)
                .map(|c0| {
                    let mut log = EventLog::default();
                    for i in c0..(c0 + CHUNK).min(end) {
                        let mut r = rng::Mt64::new(base_seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                        geo.simul_ray(params, mics, &mut log, &mut r, &dirs[i], source, energies.get(i));
                    }
                    log
                })
                .collect();
            for log in &logs {
                apply(log, hists);
            }
            start = end;
        }
    } else {
        // The RNG stream must be consumed ray by ray: trajectories are computed
        // sequentially (cheap), the logging (expensive) is done in parallel,
        // and the resulting events are applied in the original order. The next
        // batch of trajectories is computed while the previous one is logged.
        const BATCH: usize = 256;
        const CHUNK: usize = 8;
        rng::with_engine(|rng| {
            let make_paths = |rng: &mut rng::Mt64, s: usize, paths: &mut Paths<D>| {
                paths.clear();
                for i in s..(s + BATCH).min(n) {
                    geo.trace_path(params, rng, &dirs[i], source, energies.get(i), paths);
                }
            };
            let emit = |paths: &Paths<D>| -> Vec<EventLog> {
                let nr = paths.n_rays();
                (0..nr)
                    .into_par_iter()
                    .step_by(CHUNK)
                    .map(|r0| {
                        let mut log = EventLog::default();
                        for r in r0..(r0 + CHUNK).min(nr) {
                            geo.emit_path(params, mics, paths, r, &mut log);
                        }
                        log
                    })
                    .collect()
            };
            let mut current = Paths::<D>::default();
            let mut next = Paths::<D>::default();
            make_paths(rng, 0, &mut current);
            let mut s = BATCH;
            loop {
                let more = s < n;
                let logs = if more {
                    let (_, l) = rayon::join(|| make_paths(rng, s, &mut next), || emit(&current));
                    l
                } else {
                    emit(&current)
                };
                for log in &logs {
                    apply(log, hists);
                }
                if !more {
                    break;
                }
                std::mem::swap(&mut current, &mut next);
                s += BATCH;
            }
        });
    }
}
