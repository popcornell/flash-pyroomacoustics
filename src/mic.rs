//! Port of Histogram2D (common.hpp) and Microphone (microphone.hpp).

use crate::eig::*;

fn get_new_size(val: usize, cur_size: usize) -> usize {
    let mut new_size = cur_size.max(1);
    while val >= new_size {
        new_size *= 2;
    }
    new_size
}

/// Column-major 2D histogram (rows = frequency bands, cols = distance bins).
#[derive(Clone, Debug, Default)]
pub struct Histogram2D {
    pub rows: usize,
    pub cols: usize,
    pub array: Vec<f32>,
    pub counts: Vec<i32>,
}

impl Histogram2D {
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut h = Histogram2D::default();
        h.init(rows, cols);
        h
    }

    pub fn init(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.array = vec![0.0; rows * cols];
        self.counts = vec![0; rows * cols];
    }

    pub fn reset(&mut self) {
        self.array.iter_mut().for_each(|x| *x = 0.0);
        self.counts.iter_mut().for_each(|x| *x = 0);
    }

    fn resize_rows(&mut self, new_rows: usize) {
        let mut a = vec![0.0f32; new_rows * self.cols];
        let mut c = vec![0i32; new_rows * self.cols];
        let keep = self.rows.min(new_rows);
        for j in 0..self.cols {
            for i in 0..keep {
                a[j * new_rows + i] = self.array[j * self.rows + i];
                c[j * new_rows + i] = self.counts[j * self.rows + i];
            }
        }
        self.rows = new_rows;
        self.array = a;
        self.counts = c;
    }

    #[inline]
    pub fn resize_cols(&mut self, new_cols: usize) {
        self.array.resize(self.rows * new_cols, 0.0);
        self.counts.resize(self.rows * new_cols, 0);
        self.cols = new_cols;
    }

    pub fn log(&mut self, row: usize, col: usize, val: f32) {
        if row >= self.rows {
            self.resize_rows(get_new_size(row, self.rows));
        }
        if col >= self.cols {
            self.resize_cols(get_new_size(col, self.cols));
        }
        self.array[col * self.rows + row] += val;
        self.counts[col * self.rows + row] += 1;
    }

    #[inline]
    pub fn ensure_col(&mut self, col: usize) {
        if col >= self.cols {
            self.resize_cols(get_new_size(col, self.cols));
        }
    }

    #[inline]
    pub fn log_col(&mut self, col: usize, val: &[f32]) {
        self.ensure_col(col);
        let r = self.rows;
        let base = col * r;
        let n = r.min(val.len());
        for i in 0..n {
            self.array[base + i] += val[i];
            self.counts[base + i] += 1;
        }
        for i in n..r {
            self.counts[base + i] += 1;
        }
    }

    pub fn bin(&self, row: usize, col: usize) -> f32 {
        let c = self.counts[col * self.rows + row];
        if c != 0 {
            self.array[col * self.rows + row] / (c as f32)
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
pub struct MicData<const D: usize> {
    pub loc: [f32; D],
    pub n_dirs: usize,
    pub n_bands: usize,
    pub hist_resolution: f32,
    pub n_dist_bins_init: usize,
    /// direction grid (rows of the Eigen MatrixXf), only used when n_dirs > 1
    pub directions: Vec<[f32; D]>,
    pub histograms: Vec<Histogram2D>,
}

impl<const D: usize> Default for MicData<D> {
    fn default() -> Self {
        MicData {
            loc: [0.0; D],
            n_dirs: 1,
            n_bands: 1,
            hist_resolution: 1.0,
            n_dist_bins_init: 1,
            directions: vec![],
            histograms: vec![],
        }
    }
}

impl<const D: usize> MicData<D> {
    pub fn new(loc: [f32; D], n_bands: usize, hist_res: f32, max_dist_init: f32) -> Self {
        let mut m = MicData {
            loc,
            n_dirs: 1,
            n_bands,
            hist_resolution: hist_res,
            n_dist_bins_init: ((max_dist_init / hist_res) as usize) + 1,
            directions: vec![],
            histograms: vec![],
        };
        m.reset();
        m
    }

    pub fn set_directions(&mut self, dirs: Vec<[f32; D]>) {
        self.n_dirs = dirs.len();
        self.directions = dirs;
        self.reset();
    }

    pub fn make_omni(&mut self) {
        self.n_dirs = 1;
        self.directions = vec![[0.0; D]];
        self.reset();
    }

    pub fn reset_bands(&mut self, n_bands: usize) {
        self.n_bands = n_bands;
        self.histograms.resize(self.n_dirs, Histogram2D::default());
        let init = self.n_dist_bins_init;
        for h in self.histograms.iter_mut() {
            h.init(n_bands, init);
        }
    }

    pub fn reset(&mut self) {
        self.reset_bands(self.n_bands);
    }

    /// Exact nearest neighbour (squared L2 distance accumulated like nanoflann's
    /// `L2_Simple_Adaptor`).
    #[inline]
    pub fn get_dir_bin(&self, origin: &[f32; D]) -> usize {
        if self.n_dirs == 1 {
            return 0;
        }
        let dir = normalized(&sub(origin, &self.loc));
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, p) in self.directions.iter().enumerate() {
            let mut r = 0.0f32;
            for k in 0..D {
                let diff = dir[k] - p[k];
                r += diff * diff;
            }
            if r < best_d {
                best_d = r;
                best = i;
            }
        }
        best
    }

    #[inline]
    pub fn dist_bin(&self, distance: f32) -> usize {
        (distance / self.hist_resolution) as usize
    }

    #[inline]
    pub fn log_histogram(&mut self, distance: f32, energy: &[f32], origin: &[f32; D]) {
        let b = self.dist_bin(distance);
        let d = self.get_dir_bin(origin);
        self.histograms[d].log_col(b, energy);
    }
}
