//! Bounded thread pool shared by all kernels (the machine may be shared:
//! never use more threads than requested through `set_num_threads`).

use std::sync::{Arc, RwLock};

static POOL: RwLock<Option<(usize, Arc<rayon::ThreadPool>)>> = RwLock::new(None);

fn build(n: usize) -> Arc<rayon::ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n.max(1))
            .thread_name(|i| format!("pra-worker-{i}"))
            .build()
            .expect("failed to build thread pool"),
    )
}

pub fn set_num_threads(n: usize) {
    let n = n.max(1);
    let mut g = POOL.write().unwrap_or_else(|e| e.into_inner());
    if g.as_ref().map(|(m, _)| *m) != Some(n) {
        *g = Some((n, build(n)));
    }
}

pub fn num_threads() -> usize {
    POOL.read().unwrap_or_else(|e| e.into_inner()).as_ref().map(|(n, _)| *n).unwrap_or(1)
}

/// Run `f` inside the bounded pool (defaults to a single thread until
/// `set_num_threads` is called by the Python package at import time).
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let p = {
        let mut g = POOL.write().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            *g = Some((1, build(1)));
        }
        g.as_ref().unwrap().1.clone()
    };
    p.install(f)
}
