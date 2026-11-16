//! Bit-exact re-implementation of the package-wide C++ RNG:
//! `std::mt19937_64` + libstdc++'s `std::uniform_real_distribution<float>`.

use std::sync::Mutex;

const NN: usize = 312;
const MM: usize = 156;
const MATRIX_A: u64 = 0xB5026F5AA96619E9;
const UM: u64 = 0xFFFFFFFF80000000;
const LM: u64 = 0x7FFFFFFF;

#[derive(Clone)]
pub struct Mt64 {
    mt: [u64; NN],
    mti: usize,
}

impl Mt64 {
    pub fn new(seed: u64) -> Self {
        let mut s = Mt64 { mt: [0; NN], mti: NN };
        s.seed(seed);
        s
    }

    pub fn seed(&mut self, seed: u64) {
        self.mt[0] = seed;
        for i in 1..NN {
            self.mt[i] = 6364136223846793005u64
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        self.mti = NN;
    }

    fn generate(&mut self) {
        for i in 0..NN {
            let x = (self.mt[i] & UM) | (self.mt[(i + 1) % NN] & LM);
            let mut xa = x >> 1;
            if x & 1 != 0 {
                xa ^= MATRIX_A;
            }
            self.mt[i] = self.mt[(i + MM) % NN] ^ xa;
        }
        self.mti = 0;
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        if self.mti >= NN {
            self.generate();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= (x >> 29) & 0x5555555555555555;
        x ^= (x << 17) & 0x71D67FFFEDA60000;
        x ^= (x << 37) & 0xFFF7EEE000000000;
        x ^= x >> 43;
        x
    }

    /// `std::generate_canonical<float, 24>(engine)` as implemented in libstdc++ 9.
    #[inline]
    pub fn canonical_f32(&mut self) -> f32 {
        let sum = self.next_u64() as f32; // * 1.0f
        let tmp = 18446744073709551616.0f32; // 2^64
        let ret = sum / tmp;
        if ret >= 1.0 {
            // nextafter(1.0f, 0.0f)
            f32::from_bits(1.0f32.to_bits() - 1)
        } else {
            ret
        }
    }

    /// `std::uniform_real_distribution<float>(a, b)(engine)`
    #[inline]
    pub fn uniform(&mut self, a: f32, b: f32) -> f32 {
        self.canonical_f32() * (b - a) + a
    }
}

/// The global engine, default constructed with seed 5489 like `std::mt19937_64`.
pub static ENGINE: Mutex<Option<Mt64>> = Mutex::new(None);

pub fn with_engine<R>(f: impl FnOnce(&mut Mt64) -> R) -> R {
    let mut g = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        *g = Some(Mt64::new(5489));
    }
    f(g.as_mut().unwrap())
}

pub fn set_seed(seed: u64) {
    with_engine(|e| e.seed(seed));
}
