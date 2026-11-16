//! Port of libroom_src/wall.{hpp,cpp}.

use crate::eig::*;
use crate::geom::*;
use crate::rng::Mt64;

pub const TWO_PI: f32 = 2.0f32 * 3.14159265358979323846f32;
pub const PI: f32 = 3.14159265358979323846f32;
pub const HALF_PI: f32 = PI / 2.0f32;
pub const ENERGY_0: f32 = 2.0f32;

#[derive(Clone, Debug)]
pub struct WallData<const D: usize> {
    pub absorption: Vec<f32>,
    pub scatter: Vec<f32>,
    pub average_scatter: f32,
    pub does_scatter: bool,
    pub name: String,
    pub transmission: Vec<f32>,
    pub energy_reflection: Vec<f32>,
    pub normal: [f32; D],
    /// corners as columns
    pub corners: Vec<[f32; D]>,
    pub origin: [f32; D],
    /// basis columns (only meaningful in 3D)
    pub basis: [[f32; D]; 2],
    /// flat corners as columns (only meaningful in 3D)
    pub flat_corners: Vec<[f32; 2]>,
}

#[derive(Debug)]
pub struct WallError(pub String);

impl<const D: usize> WallData<D> {
    fn init(&mut self) -> Result<(), WallError> {
        self.energy_reflection = self.absorption.iter().map(|a| 1.0f32 - a).collect();
        self.transmission = self.energy_reflection.iter().map(|e| e.sqrt()).collect();
        self.average_scatter = if self.scatter.is_empty() { f32::NAN } else { vmean(&self.scatter) };
        self.does_scatter = !self.scatter.is_empty() && vmax(&self.scatter) > 0.0;
        if self.absorption.len() != self.scatter.len() {
            return Err(WallError(
                "The number of absorption and scattering coefficients is different".into(),
            ));
        }
        Ok(())
    }

    pub fn new(
        corners: Vec<[f32; D]>,
        absorption: Vec<f32>,
        scatter: Vec<f32>,
        name: String,
    ) -> Result<Self, WallError> {
        let mut w = WallData {
            absorption,
            scatter,
            average_scatter: 0.0,
            does_scatter: false,
            name,
            transmission: vec![],
            energy_reflection: vec![],
            normal: [0.0; D],
            corners,
            origin: [0.0; D],
            basis: [[0.0; D]; 2],
            flat_corners: vec![],
        };
        w.init()?;
        if w.corners.is_empty() {
            return Err(WallError("A wall needs at least one corner".into()));
        }
        w.origin = w.corners[0];
        if D == 2 {
            if w.corners.len() < 2 {
                return Err(WallError("A 2D wall needs two corners".into()));
            }
            let c = &w.corners;
            let mut n = [0.0f32; D];
            n[0] = c[1][1] - c[0][1];
            n[1] = c[0][0] - c[1][0];
            w.normal = normalized(&n);
            // The original never initializes the 2D basis (it reads uninitialized
            // memory when sampling scattered directions). We use the wall tangent,
            // which makes 2D Lambertian scattering well defined.
            w.basis[0][0] = -w.normal[1];
            w.basis[0][1] = w.normal[0];
        } else {
            if w.corners.len() < 3 {
                return Err(WallError("The corners of the wall do not lie in a plane".into()));
            }
            let a: Vec<[f32; 3]> = w
                .corners
                .iter()
                .map(|c| {
                    let d = sub(c, &w.origin);
                    [d[0], d[1], d[2]]
                })
                .collect();
            let (u, sv) = jacobi_svd_3xn_u(&a);
            if sv[2] > eps() {
                return Err(WallError("The corners of the wall do not lie in a plane".into()));
            }
            let mut b0 = [0.0f32; 3];
            let mut b1 = [0.0f32; 3];
            for r in 0..3 {
                b0[r] = u[r][0];
                b1[r] = u[r][1];
            }
            let mut flat: Vec<[f32; 2]> = a.iter().map(|v| [dot_tree(&b0, v), dot_tree(&b1, v)]).collect();
            let area = area_2d_polygon(&flat);
            if area < 0.0 {
                std::mem::swap(&mut b0, &mut b1);
                for f in flat.iter_mut() {
                    f.swap(0, 1);
                }
            }
            let n = cross(&b0, &b1);
            let mut bb = [[0.0f32; D]; 2];
            let mut nn = [0.0f32; D];
            for r in 0..3 {
                bb[0][r] = b0[r];
                bb[1][r] = b1[r];
                nn[r] = n[r];
            }
            w.basis = bb;
            w.normal = nn;
            w.flat_corners = flat;
        }
        Ok(w)
    }

    #[inline]
    pub fn n_bands(&self) -> usize {
        self.transmission.len()
    }

    pub fn broadcast_bands_to(&mut self, new_n_bands: usize) -> Result<(), WallError> {
        let n = self.n_bands();
        if n == 1 || new_n_bands == 1 {
            let a = self.absorption[0];
            let s = self.scatter[0];
            let t = self.transmission[0];
            let e = self.energy_reflection[0];
            self.absorption = vec![a; new_n_bands];
            self.scatter = vec![s; new_n_bands];
            self.transmission = vec![t; new_n_bands];
            self.energy_reflection = vec![e; new_n_bands];
            Ok(())
        } else {
            Err(WallError("Incompatible bands broadcasting size.".into()))
        }
    }

    pub fn area(&self) -> f32 {
        if D == 2 {
            norm(&sub(&self.corners[1], &self.corners[0]))
        } else {
            area_2d_polygon(&self.flat_corners)
        }
    }

    /// Wall::intersection: may write into `out` even when no intersection is reported
    /// (exactly like the original).
    #[inline]
    pub fn intersection(&self, p1: &[f32; D], p2: &[f32; D], out: &mut [f32; D]) -> i32 {
        if D == 2 {
            let a1 = [p1[0], p1[1]];
            let a2 = [p2[0], p2[1]];
            let b1 = [self.corners[0][0], self.corners[0][1]];
            let b2 = [self.corners[1][0], self.corners[1][1]];
            let mut o = [out[0], out[1]];
            let r = intersection_2d_segments(&a1, &a2, &b1, &b2, &mut o);
            out[0] = o[0];
            out[1] = o[1];
            r
        } else {
            let a1 = [p1[0], p1[1], p1[2]];
            let a2 = [p2[0], p2[1], p2[2]];
            let org = [self.origin[0], self.origin[1], self.origin[2]];
            let nrm = [self.normal[0], self.normal[1], self.normal[2]];
            let mut o = [out[0], out[1], out[2]];
            let ret1 = intersection_3d_segment_plane(&a1, &a2, &org, &nrm, &mut o);
            out[0] = o[0];
            out[1] = o[1];
            out[2] = o[2];
            if ret1 == -1 {
                return WALL_ISECT_NONE;
            }
            let mut ret = 0;
            if ret1 == 1 {
                ret = WALL_ISECT_VALID_ENDPT;
            }
            let d = sub(&o, &org);
            let b0 = [self.basis[0][0], self.basis[0][1], self.basis[0][2]];
            let b1 = [self.basis[1][0], self.basis[1][1], self.basis[1][2]];
            let flat = [dot_tree(&b0, &d), dot_tree(&b1, &d)];
            let ret2 = is_inside_2d_polygon(&flat, &self.flat_corners);
            if ret2 < 0 {
                return WALL_ISECT_NONE;
            }
            if ret2 == 1 {
                ret |= WALL_ISECT_VALID_BNDRY;
            }
            ret
        }
    }

    #[inline]
    pub fn intersects(&self, p1: &[f32; D], p2: &[f32; D]) -> i32 {
        let mut v = [0.0f32; D];
        self.intersection(p1, p2, &mut v)
    }

    #[inline]
    pub fn reflect(&self, p: &[f32; D], out: &mut [f32; D]) -> i32 {
        let d = dot(&self.normal, &sub(&self.origin, p));
        let k = 2.0f32 * d;
        for i in 0..D {
            out[i] = p[i] + k * self.normal[i];
        }
        let e = eps();
        if d > e {
            1
        } else if d < -e {
            -1
        } else {
            0
        }
    }

    #[inline]
    pub fn side(&self, p: &[f32; D]) -> i32 {
        let ip = dot(&sub(p, &self.origin), &self.normal);
        let e = eps();
        if ip > e {
            1
        } else if ip < -e {
            -1
        } else {
            0
        }
    }

    pub fn same_as(&self, that: &WallData<D>) -> bool {
        if self.corners.len() != that.corners.len() {
            return false;
        }
        let mut diffs = Vec::with_capacity(D * self.corners.len());
        for (a, b) in self.corners.iter().zip(&that.corners) {
            for i in 0..D {
                diffs.push((a[i] - b[i]).abs());
            }
        }
        vsum(&diffs) as f64 == 0.0
    }

    #[inline]
    pub fn normal_reflect_dir(&self, incident: &[f32; D]) -> [f32; D] {
        let d = dot(incident, &self.normal);
        let mut r = [0.0f32; D];
        for i in 0..D {
            r[i] = incident[i] - (self.normal[i] * 2.0f32) * d;
        }
        r
    }

    pub fn normal_reflect(&self, start: &[f32; D], hit_point: &[f32; D], length: f32) -> [f32; D] {
        let incident = normalized(&sub(hit_point, start));
        let d = dot(&incident, &self.normal);
        let mut r = [0.0f32; D];
        for i in 0..D {
            r[i] = hit_point[i] + length * (incident[i] - (self.normal[i] * 2.0f32) * d);
        }
        r
    }

    #[inline]
    pub fn cosine_angle(&self, p: &[f32; D]) -> f32 {
        dot(p, &self.normal) / norm(p)
    }

    #[inline]
    pub fn sample_lambertian_reflection(&self, rng: &mut Mt64) -> [f32; D] {
        let u1 = rng.uniform(0.0, 1.0);
        let u2 = rng.uniform(0.0, 1.0);
        let r = u1.sqrt();
        let phi = TWO_PI * u2;
        let x_l = r * phi.cos();
        let y_l = r * phi.sin();
        let z_l = (0.0f64.max(1.0f64 - u1 as f64)).sqrt() as f32;
        let mut out = [0.0f32; D];
        for i in 0..D {
            out[i] = (self.basis[0][i] * x_l + self.basis[1][i] * y_l) + z_l * self.normal[i];
        }
        out
    }
}
