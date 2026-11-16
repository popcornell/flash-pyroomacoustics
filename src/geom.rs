//! Geometry routines (port of libroom_src/geometry.cpp).

use crate::eig::*;
use std::sync::atomic::{AtomicU32, Ordering};

static EPS_BITS: AtomicU32 = AtomicU32::new(0x3727c5ac); // 1e-5f32

#[inline(always)]
pub fn eps() -> f32 {
    f32::from_bits(EPS_BITS.load(Ordering::Relaxed))
}

pub fn set_eps(e: f32) {
    EPS_BITS.store(e.to_bits(), Ordering::Relaxed);
}

pub const WALL_ISECT_NONE: i32 = -1;
pub const WALL_ISECT_VALID: i32 = 0;
pub const WALL_ISECT_VALID_ENDPT: i32 = 1;
pub const WALL_ISECT_VALID_BNDRY: i32 = 2;

pub type V2 = [f32; 2];
pub type V3 = [f32; 3];

#[inline]
pub fn ccw3p(p1: &V2, p2: &V2, p3: &V2) -> i32 {
    let d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]);
    let e = eps();
    if d < e && d > -e {
        0
    } else if d > 0.0 {
        1
    } else {
        -1
    }
}

#[inline]
pub fn check_intersection_2d_segments(a1: &V2, a2: &V2, b1: &V2, b2: &V2) -> i32 {
    let a1a2b1 = ccw3p(a1, a2, b1);
    let a1a2b2 = ccw3p(a1, a2, b2);
    if a1a2b1 == a1a2b2 {
        return -1;
    }
    let b1b2a1 = ccw3p(b1, b2, a1);
    let b1b2a2 = ccw3p(b1, b2, a2);
    if b1b2a1 == b1b2a2 {
        return -1;
    }
    let mut ret = 0;
    if b1b2a1 == 0 || b1b2a2 == 0 {
        ret |= 1;
    }
    if a1a2b1 == 0 || a1a2b2 == 0 {
        ret |= 2;
    }
    ret
}

/// Returns (ret, intersection). `intersection` is only written when ret >= 0.
#[inline]
pub fn intersection_2d_segments(a1: &V2, a2: &V2, b1: &V2, b2: &V2, out: &mut V2) -> i32 {
    let ret = check_intersection_2d_segments(a1, a2, b1, b2);
    if ret < 0 {
        return ret;
    }
    let normal = [a1[1] - a2[1], a2[0] - a1[0]];
    let db = sub(b2, b1);
    let denom = dot(&normal, &db);
    if denom.abs() < eps() {
        return -1;
    }
    let num = dot(&normal, &sub(a1, b1));
    let r = num / denom;
    out[0] = r * db[0] + b1[0];
    out[1] = r * db[1] + b1[1];
    ret
}

#[inline]
pub fn intersection_3d_segment_plane(a1: &V3, a2: &V3, p: &V3, normal: &V3, out: &mut V3) -> i32 {
    let u = sub(a2, a1);
    let denom = dot(normal, &u);
    let e = eps();
    if denom.abs() > e {
        let w = sub(a1, p);
        let num = dot(&neg(normal), &w);
        let s = num / denom;
        if 0.0 - e <= s && s <= 1.0 + e {
            for i in 0..3 {
                out[i] = s * u[i] + a1[i];
            }
            if s.abs() < e || (s - 1.0).abs() < e {
                return 1;
            } else {
                return 0;
            }
        }
    }
    -1
}

#[inline]
fn on_segment(c1: &V2, c2: &V2, p: &V2) -> bool {
    let x_down = c1[0].min(c2[0]);
    let x_up = c1[0].max(c2[0]);
    let y_down = c1[1].min(c2[1]);
    let y_up = c1[1].max(c2[1]);
    x_down <= p[0] && p[0] <= x_up && y_down <= p[1] && p[1] <= y_up
}

#[inline]
fn is_left(p0: &V2, p1: &V2, p2: &V2) -> i32 {
    let test_value = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
    if test_value.abs() < eps() {
        0
    } else if test_value > 0.0 {
        1
    } else {
        -1
    }
}

/// `corners` is a list of 2D points (columns of the Eigen 2xN matrix).
pub fn is_inside_2d_polygon(p: &V2, corners: &[V2]) -> i32 {
    let n = corners.len();
    let mut wn = 0i32;
    if n == 0 {
        return -1;
    }
    let mut j = n - 1;
    for i in 0..n {
        let cj = &corners[j];
        let ci = &corners[i];
        if ccw3p(cj, ci, p) == 0 && on_segment(cj, ci, p) {
            return 1;
        }
        if cj[1] <= p[1] {
            if ci[1] > p[1] && is_left(cj, ci, p) > 0 {
                wn += 1;
            }
        } else if ci[1] <= p[1] && is_left(cj, ci, p) < 0 {
            wn -= 1;
        }
        j = i;
    }
    if wn == 0 {
        -1
    } else {
        0
    }
}

pub fn area_2d_polygon(corners: &[V2]) -> f32 {
    let n = corners.len();
    let mut a = 0.0f32;
    for c1 in 0..n {
        let c2 = if c1 == n - 1 { 0 } else { c1 + 1 };
        let base = (0.5f64 * ((corners[c2][1] + corners[c1][1]) as f64)) as f32;
        let height = corners[c2][0] - corners[c1][0];
        a -= height * base;
    }
    a
}

/// Normalize a dynamic-size Eigen vector (`VectorXf::normalized()`).
fn vnormalized(v: &[f32]) -> Vec<f32> {
    let n = vsqnorm(v);
    if n > 0.0 {
        let s = n.sqrt();
        v.iter().map(|x| x / s).collect()
    } else {
        v.to_vec()
    }
}

pub fn cos_angle_between(v1: &[f32], v2: &[f32]) -> f32 {
    let a = vnormalized(v1);
    let b = vnormalized(v2);
    let d = vdot(&a, &b) as f64;
    let c = if d < -1.0 {
        -1.0
    } else if d > 1.0 {
        1.0
    } else {
        d
    };
    c as f32
}

pub fn dist_line_point(start: &[f32], end: &[f32], point: &[f32]) -> f32 {
    let diff: Vec<f32> = end.iter().zip(start).map(|(e, s)| e - s).collect();
    let unit = vnormalized(&diff);
    let v: Vec<f32> = point.iter().zip(start).map(|(p, s)| p - s).collect();
    let proj = vdot(&v, &unit);
    let w: Vec<f32> = v.iter().zip(&unit).map(|(a, u)| a - proj * u).collect();
    vsqnorm(&w).sqrt()
}
