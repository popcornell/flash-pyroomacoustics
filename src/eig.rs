//! Bit-exact emulation of the Eigen 5.0.1 operations used by the original
//! C++ libroom (compiled with GCC, -O3, SSE2, no FMA).
//!
//! Floating point addition is not associative, so to reproduce the exact
//! same results as the original extension we must reproduce the order in
//! which Eigen accumulates sums. The orders below were determined by reading
//! the Eigen sources and verified empirically (see `probes/` in the repo).
//!
//! * Fixed-size `squaredNorm()`/`norm()` and fixed-size matrix-vector
//!   products are completely unrolled as a binary tree: `a0 + (a1 + a2)`.
//! * Fixed-size `dot()` and `u.adjoint() * v` are sequential: `(a0 + a1) + a2`.
//! * Dynamic-size reductions use the "linear vectorized" traversal with
//!   4-wide SSE packets (see [`vsum`]).

/// Horizontal sum of an SSE packet, as done by Eigen's `predux<Packet4f>`.
#[inline(always)]
pub fn predux(p: [f32; 4]) -> f32 {
    (p[0] + p[2]) + (p[1] + p[3])
}

/// Sum of a dynamic-size expression as performed by Eigen's
/// `redux_impl<..., LinearVectorizedTraversal, NoUnrolling>` with
/// `alignedStart == 0` (which is the case for all the expressions we
/// emulate: either heap-allocated aligned storage or non-direct-access
/// expressions).
#[inline]
pub fn vsum_iter<F: Fn(usize) -> f32>(n: usize, x: F) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let aligned_size = (n / 4) * 4;
    if aligned_size == 0 {
        let mut r = x(0);
        for i in 1..n {
            r += x(i);
        }
        return r;
    }
    let aligned_size2 = (n / 8) * 8;
    let mut p0 = [x(0), x(1), x(2), x(3)];
    if aligned_size > 4 {
        let mut p1 = [x(4), x(5), x(6), x(7)];
        let mut idx = 8;
        while idx < aligned_size2 {
            for l in 0..4 {
                p0[l] += x(idx + l);
                p1[l] += x(idx + 4 + l);
            }
            idx += 8;
        }
        for l in 0..4 {
            p0[l] += p1[l];
        }
        if aligned_size > aligned_size2 {
            for l in 0..4 {
                p0[l] += x(aligned_size2 + l);
            }
        }
    }
    let mut r = predux(p0);
    for i in aligned_size..n {
        r += x(i);
    }
    r
}

#[inline]
pub fn vsum(x: &[f32]) -> f32 {
    vsum_iter(x.len(), |i| x[i])
}

/// Squared norm of a dynamic-size vector.
#[inline]
pub fn vsqnorm(x: &[f32]) -> f32 {
    vsum_iter(x.len(), |i| x[i] * x[i])
}

/// Dot product of two dynamic-size vectors.
#[inline]
pub fn vdot(a: &[f32], b: &[f32]) -> f32 {
    vsum_iter(a.len(), |i| a[i] * b[i])
}

/// Mean of a dynamic-size array (`ArrayXf::mean()`).
#[inline]
pub fn vmean(x: &[f32]) -> f32 {
    vsum(x) / (x.len() as f32)
}

/// Maximum coefficient (`maxCoeff()`), order independent for non-NaN input.
#[inline]
pub fn vmax(x: &[f32]) -> f32 {
    let mut m = x[0];
    for &v in &x[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

/// One row of Eigen's row-major general matrix-vector product kernel
/// (`general_matrix_vector_product<..., RowMajor, ...>`), SSE, no FMA.
#[inline]
pub fn gemv_row_dot(n: usize, a: impl Fn(usize) -> f32, b: impl Fn(usize) -> f32) -> f32 {
    let full = (n / 4) * 4;
    let mut c = [0.0f32; 4];
    let mut j = 0;
    while j < full {
        for l in 0..4 {
            c[l] = a(j + l) * b(j + l) + c[l];
        }
        j += 4;
    }
    let mut cc = predux(c);
    for j in full..n {
        cc += a(j) * b(j);
    }
    0.0 + cc
}

// ---------------------------------------------------------------------------
// Small fixed-size vector helpers (D = 2 or 3)
// ---------------------------------------------------------------------------

/// Tree-ordered sum used by fixed-size `squaredNorm` and fixed-size
/// coefficient-based matrix products.
#[inline(always)]
pub fn tree_sum<const D: usize>(t: [f32; D]) -> f32 {
    match D {
        1 => t[0],
        2 => t[0] + t[1],
        3 => t[0] + (t[1] + t[2]),
        _ => {
            // generic binary splitting
            fn rec(t: &[f32]) -> f32 {
                if t.len() == 1 {
                    t[0]
                } else {
                    let h = t.len() / 2;
                    rec(&t[..h]) + rec(&t[h..])
                }
            }
            rec(&t)
        }
    }
}

/// Sequential sum used by fixed-size `dot()` and `adjoint() * v`.
#[inline(always)]
pub fn seq_sum<const D: usize>(t: [f32; D]) -> f32 {
    let mut r = t[0];
    for i in 1..D {
        r += t[i];
    }
    r
}

#[inline(always)]
pub fn sub<const D: usize>(a: &[f32; D], b: &[f32; D]) -> [f32; D] {
    let mut r = [0.0; D];
    for i in 0..D {
        r[i] = a[i] - b[i];
    }
    r
}

#[inline(always)]
pub fn add<const D: usize>(a: &[f32; D], b: &[f32; D]) -> [f32; D] {
    let mut r = [0.0; D];
    for i in 0..D {
        r[i] = a[i] + b[i];
    }
    r
}

/// `v * s` (also equals `s * v` bitwise)
#[inline(always)]
pub fn scale<const D: usize>(v: &[f32; D], s: f32) -> [f32; D] {
    let mut r = [0.0; D];
    for i in 0..D {
        r[i] = v[i] * s;
    }
    r
}

#[inline(always)]
pub fn neg<const D: usize>(v: &[f32; D]) -> [f32; D] {
    let mut r = [0.0; D];
    for i in 0..D {
        r[i] = -v[i];
    }
    r
}

/// Fixed-size `a.dot(b)` and `a.adjoint() * b`.
#[inline(always)]
pub fn dot<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut t = [0.0; D];
    for i in 0..D {
        t[i] = a[i] * b[i];
    }
    seq_sum(t)
}

/// Fixed-size row of a coefficient-based product, e.g. `basis.adjoint() * v`.
#[inline(always)]
pub fn dot_tree<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut t = [0.0; D];
    for i in 0..D {
        t[i] = a[i] * b[i];
    }
    tree_sum(t)
}

#[inline(always)]
pub fn sqnorm<const D: usize>(a: &[f32; D]) -> f32 {
    let mut t = [0.0; D];
    for i in 0..D {
        t[i] = a[i] * a[i];
    }
    tree_sum(t)
}

#[inline(always)]
pub fn norm<const D: usize>(a: &[f32; D]) -> f32 {
    sqnorm(a).sqrt()
}

/// `normalized()`: returns `v / sqrt(squaredNorm)` when the norm is positive.
#[inline(always)]
pub fn normalized<const D: usize>(a: &[f32; D]) -> [f32; D] {
    let n = sqnorm(a);
    if n > 0.0 {
        let s = n.sqrt();
        let mut r = [0.0; D];
        for i in 0..D {
            r[i] = a[i] / s;
        }
        r
    } else {
        *a
    }
}

#[inline(always)]
pub fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// JacobiSVD<Matrix<float, 3, Dynamic>> with ComputeThinU
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct JRot {
    c: f32,
    s: f32,
}

impl JRot {
    fn transpose(self) -> JRot {
        JRot { c: self.c, s: -self.s }
    }
    fn mul(self, o: JRot) -> JRot {
        JRot {
            c: self.c * o.c - self.s * o.s,
            s: self.c * o.s + self.s * o.c,
        }
    }
    /// JacobiRotation::makeJacobi(x, y, z)
    fn make_jacobi(x: f32, y: f32, z: f32) -> JRot {
        let deno = 2.0f32 * y.abs();
        if deno < f32::MIN_POSITIVE {
            JRot { c: 1.0, s: 0.0 }
        } else {
            let tau = (x - z) / deno;
            let w = (tau * tau + 1.0f32).sqrt();
            let t = if tau > 0.0 { 1.0f32 / (tau + w) } else { 1.0f32 / (tau - w) };
            let sign_t = if t > 0.0 { 1.0f32 } else { -1.0f32 };
            let n = 1.0f32 / (t * t + 1.0f32).sqrt();
            let s = (((-sign_t) * (y / y.abs())) * t.abs()) * n;
            JRot { c: n, s }
        }
    }
}

/// Apply a rotation to two "vectors" x and y in place:
/// x' = c x + s y ; y' = -s x + c y
#[inline(always)]
fn rot_pair(x: &mut f32, y: &mut f32, j: JRot) {
    let xi = *x;
    let yi = *y;
    *x = j.c * xi + j.s * yi;
    *y = -j.s * xi + j.c * yi;
}

type M3 = [[f32; 3]; 3]; // row-major m[row][col]

fn apply_on_the_left(m: &mut M3, p: usize, q: usize, j: JRot) {
    for c in 0..3 {
        let (mut x, mut y) = (m[p][c], m[q][c]);
        rot_pair(&mut x, &mut y, j);
        m[p][c] = x;
        m[q][c] = y;
    }
}

/// `applyOnTheRight(p, q, j)` which applies `j.transpose()` to columns p, q
fn apply_on_the_right(m: &mut M3, p: usize, q: usize, j: JRot) {
    let jt = j.transpose();
    for r in 0..3 {
        let (mut x, mut y) = (m[r][p], m[r][q]);
        rot_pair(&mut x, &mut y, jt);
        m[r][p] = x;
        m[r][q] = y;
    }
}

fn real_2x2_jacobi_svd(w: &M3, p: usize, q: usize) -> (JRot, JRot) {
    let mut m = [[w[p][p], w[p][q]], [w[q][p], w[q][q]]];
    let t = m[0][0] + m[1][1];
    let d = m[1][0] - m[0][1];
    let rot1 = if d.abs() < f32::MIN_POSITIVE {
        JRot { c: 1.0, s: 0.0 }
    } else {
        let u = t / d;
        let tmp = (1.0f32 + u * u).sqrt();
        JRot { s: 1.0f32 / tmp, c: u / tmp }
    };
    // m.applyOnTheLeft(0, 1, rot1)
    for c in 0..2 {
        let (mut x, mut y) = (m[0][c], m[1][c]);
        rot_pair(&mut x, &mut y, rot1);
        m[0][c] = x;
        m[1][c] = y;
    }
    let j_right = JRot::make_jacobi(m[0][0], m[0][1], m[1][1]);
    let j_left = rot1.mul(j_right.transpose());
    (j_left, j_right)
}

/// Column-pivoting Householder QR of an n x 3 column-major matrix (`cols[c][r]`).
/// Returns the 3x3 upper triangular factor R (row-major) and the column
/// permutation indices.
fn colpiv_householder_qr(mut cols: [Vec<f32>; 3]) -> (M3, [usize; 3]) {
    let rows = cols[0].len();
    let ncols = 3usize;
    let size = rows.min(ncols);

    let mut norms_direct = [0.0f32; 3];
    let mut norms_updated = [0.0f32; 3];
    for k in 0..ncols {
        norms_direct[k] = vsqnorm(&cols[k]).sqrt();
        norms_updated[k] = norms_direct[k];
    }
    let eps = f32::EPSILON;
    let norm_downdate_threshold = eps.sqrt();
    let mut transpositions = [0usize; 3];

    for k in 0..size {
        // biggest column (first occurrence)
        let mut biggest = k;
        let mut bval = norms_updated[k];
        for j in k + 1..ncols {
            if norms_updated[j] > bval {
                bval = norms_updated[j];
                biggest = j;
            }
        }
        transpositions[k] = biggest;
        if k != biggest {
            cols.swap(k, biggest);
            norms_updated.swap(k, biggest);
            norms_direct.swap(k, biggest);
        }

        // makeHouseholderInPlace on cols[k][k..]
        let tau;
        let beta;
        {
            let col = &mut cols[k];
            let c0 = col[k];
            let tail_sq = if rows - k == 1 { 0.0 } else { vsqnorm(&col[k + 1..]) };
            let tol = f32::MIN_POSITIVE;
            if tail_sq <= tol {
                tau = 0.0f32;
                beta = c0;
                for v in col[k + 1..].iter_mut() {
                    *v = 0.0;
                }
            } else {
                let mut b = (c0 * c0 + tail_sq).sqrt();
                if c0 >= 0.0 {
                    b = -b;
                }
                beta = b;
                let denom = c0 - b;
                for v in col[k + 1..].iter_mut() {
                    *v = *v / denom;
                }
                tau = (b - c0) / b;
            }
            col[k] = beta;
        }

        // applyHouseholderOnTheLeft to bottomRightCorner(rows - k, cols - k - 1)
        let nb = ncols - k - 1;
        let brows = rows - k;
        if nb > 0 {
            if brows == 1 {
                for j in k + 1..ncols {
                    cols[j][k] *= 1.0f32 - tau;
                }
            } else if tau != 0.0 {
                let ess: Vec<f32> = cols[k][k + 1..].to_vec();
                let m = ess.len();
                let mut tmp = vec![0.0f32; nb];
                if nb == 1 {
                    // inner product fallback: dst += alpha * dot
                    let b = &cols[k + 1][k + 1..];
                    tmp[0] = 0.0 + vdot(&ess, b);
                } else {
                    for jj in 0..nb {
                        let b = &cols[k + 1 + jj][k + 1..];
                        tmp[jj] = gemv_row_dot(m, |i| b[i], |i| ess[i]);
                    }
                }
                for jj in 0..nb {
                    tmp[jj] += cols[k + 1 + jj][k];
                }
                for jj in 0..nb {
                    cols[k + 1 + jj][k] -= tau * tmp[jj];
                }
                let tess: Vec<f32> = ess.iter().map(|&e| tau * e).collect();
                for jj in 0..nb {
                    let t = tmp[jj];
                    let c = &mut cols[k + 1 + jj];
                    for i in 0..m {
                        c[k + 1 + i] -= t * tess[i];
                    }
                }
            }
        }

        // norm downdate
        for j in k + 1..ncols {
            if norms_updated[j] != 0.0 {
                let mut temp = cols[j][k].abs() / norms_updated[j];
                temp = (1.0f32 + temp) * (1.0f32 - temp);
                if temp < 0.0 {
                    temp = 0.0;
                }
                let r = norms_updated[j] / norms_direct[j];
                let temp2 = temp * (r * r);
                if temp2 <= norm_downdate_threshold {
                    norms_direct[j] = vsqnorm(&cols[j][k + 1..]).sqrt();
                    norms_updated[j] = norms_direct[j];
                } else {
                    norms_updated[j] *= temp.sqrt();
                }
            }
        }
    }

    let mut indices = [0usize, 1, 2];
    for k in 0..size {
        indices.swap(k, transpositions[k]);
    }

    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            r[i][j] = cols[j][i];
        }
    }
    (r, indices)
}

/// Computes the thin U factor and singular values of the 3 x N matrix `a`
/// given as columns (`a[j]` is the j-th column) exactly as
/// `Eigen::JacobiSVD<Eigen::Matrix<float,3,Dynamic>>(a, ComputeThinU)`.
///
/// Requires N >= 3. Returns (U as row-major 3x3, singular values).
pub fn jacobi_svd_3xn_u(a: &[[f32; 3]]) -> ([[f32; 3]; 3], [f32; 3]) {
    let n = a.len();
    assert!(n >= 3);
    // scale = max |a_ij| (PropagateNaN)
    let mut scale = 0.0f32;
    let mut has_nan = false;
    for col in a {
        for &v in col {
            if v.is_nan() {
                has_nan = true;
            }
            let av = v.abs();
            if av > scale {
                scale = av;
            }
        }
    }
    if has_nan || !scale.is_finite() {
        // Eigen sets info=InvalidInput and returns without touching U / singular values.
        // The original code would then read uninitialized memory; we return NaNs.
        return ([[f32::NAN; 3]; 3], [f32::NAN; 3]);
    }
    if scale == 0.0 {
        scale = 1.0;
    }

    let mut w: M3;
    let mut u: M3;
    if n == 3 {
        w = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                w[i][j] = a[j][i] / scale;
            }
        }
        u = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    } else {
        // QR of the adjoint (n x 3), column c = coordinate c of all corners
        let cols: [Vec<f32>; 3] = [
            a.iter().map(|v| v[0] / scale).collect(),
            a.iter().map(|v| v[1] / scale).collect(),
            a.iter().map(|v| v[2] / scale).collect(),
        ];
        let (r, indices) = colpiv_householder_qr(cols);
        w = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..=i {
                w[i][j] = r[j][i];
            }
        }
        u = [[0.0; 3]; 3];
        for i in 0..3 {
            u[indices[i]][i] = 1.0;
        }
    }

    let precision = 2.0f32 * f32::EPSILON;
    let consider_as_zero = f32::MIN_POSITIVE;
    let mut max_diag = w[0][0].abs();
    for i in 1..3 {
        let v = w[i][i].abs();
        if v > max_diag {
            max_diag = v;
        }
    }
    let maxi = |a: f32, b: f32| if a < b { b } else { a };

    let mut finished = false;
    while !finished {
        finished = true;
        for p in 1..3 {
            for q in 0..p {
                let threshold = maxi(consider_as_zero, precision * max_diag);
                if w[p][q].abs() > threshold || w[q][p].abs() > threshold {
                    finished = false;
                    let (j_left, j_right) = real_2x2_jacobi_svd(&w, p, q);
                    apply_on_the_left(&mut w, p, q, j_left);
                    apply_on_the_right(&mut u, p, q, j_left.transpose());
                    apply_on_the_right(&mut w, p, q, j_right);
                    max_diag = maxi(max_diag, maxi(w[p][p].abs(), w[q][q].abs()));
                }
            }
        }
    }

    let mut sv = [0.0f32; 3];
    for i in 0..3 {
        let a = w[i][i];
        sv[i] = a.abs();
        if a < 0.0 {
            for r in 0..3 {
                u[r][i] = -u[r][i];
            }
        }
    }
    for i in 0..3 {
        sv[i] *= scale;
    }

    for i in 0..3 {
        // maxCoeff(&pos) over tail, first occurrence
        let mut pos = 0usize;
        let mut mx = sv[i];
        for k in i + 1..3 {
            if sv[k] > mx {
                mx = sv[k];
                pos = k - i;
            }
        }
        if mx == 0.0 {
            break;
        }
        if pos != 0 {
            let pos = pos + i;
            sv.swap(i, pos);
            for r in 0..3 {
                let t = u[r][pos];
                u[r][pos] = u[r][i];
                u[r][i] = t;
            }
        }
    }
    (u, sv)
}
