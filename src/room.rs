//! Port of libroom_src/room.{hpp,cpp}: image source model and ray tracing.

use crate::eig::*;
use crate::geom::*;
use crate::mic::MicData;
use crate::rng::Mt64;
use crate::wall::*;

#[derive(Debug)]
pub struct RoomError(pub String);

impl From<WallError> for RoomError {
    fn from(e: WallError) -> Self {
        RoomError(e.0)
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub sound_speed: f32,
    pub ism_order: i32,
    pub energy_thres: f32,
    pub time_thres: f32,
    pub mic_radius: f32,
    pub mic_radius_sq: f64,
    pub mic_hist_res: f32,
    pub is_hybrid_sim: bool,
}

impl Params {
    pub fn new(
        sound_speed: f32,
        ism_order: i32,
        energy_thres: f32,
        time_thres: f32,
        mic_radius: f32,
        mic_hist_res: f32,
        is_hybrid_sim: bool,
    ) -> Self {
        Params {
            sound_speed,
            ism_order,
            energy_thres,
            time_thres,
            mic_radius,
            mic_radius_sq: (mic_radius * mic_radius) as f64,
            mic_hist_res,
            is_hybrid_sim,
        }
    }
}

/// Results of the image source model (column-major data like Eigen)
#[derive(Clone, Debug, Default)]
pub struct IsmResults<const D: usize> {
    pub sources: Vec<[f32; D]>,
    pub gen_walls: Vec<i32>,
    pub orders: Vec<i32>,
    pub orders_xyz: Vec<[i32; D]>,
    pub source_directions: Vec<[f32; D]>,
    /// column-major (n_bands x n_sources)
    pub attenuations: Vec<f32>,
    /// column-major (n_mics x n_sources)
    pub visible_mics: Vec<bool>,
    pub n_sources: usize,
    pub n_bands: usize,
    pub n_mics: usize,
}

#[derive(Clone, Debug)]
pub struct Geometry<const D: usize> {
    pub walls: Vec<WallData<D>>,
    pub obstructing_walls: Vec<i32>,
    pub is_shoebox: bool,
    pub shoebox_size: [f32; D],
    /// shoebox absorption/scattering: n_bands x 2D (row-major [band][wall])
    pub shoebox_absorption: Vec<Vec<f32>>,
    pub shoebox_scattering: Vec<Vec<f32>>,
    pub n_bands: usize,
    pub max_dist: f32,
}

fn cols_from_rowmajor<const D: usize>(vals: &[f32]) -> Vec<[f32; D]> {
    let n = vals.len() / D;
    (0..n)
        .map(|j| {
            let mut c = [0.0f32; D];
            for i in 0..D {
                c[i] = vals[i * n + j];
            }
            c
        })
        .collect()
}

pub fn make_shoebox_walls<const D: usize>(
    rs: &[f32; D],
    abs: &[Vec<f32>],
    scat: &[Vec<f32>],
) -> Result<Vec<WallData<D>>, RoomError> {
    let col = |m: &[Vec<f32>], c: usize| -> Vec<f32> { m.iter().map(|row| row[c]).collect() };
    let mut walls = vec![];
    if D == 2 {
        let (r0, r1) = (rs[0], rs[1]);
        let defs: [(&[f32], usize, &str); 4] = [
            (&[0.0, r0, 0.0, 0.0], 2, "south"),
            (&[r0, r0, 0.0, r1], 1, "east"),
            (&[r0, 0.0, r1, r1], 3, "north"),
            (&[0.0, 0.0, r1, 0.0], 0, "west"),
        ];
        for (vals, c, name) in defs.iter() {
            walls.push(WallData::<D>::new(
                cols_from_rowmajor::<D>(vals),
                col(abs, *c),
                col(scat, *c),
                name.to_string(),
            )?);
        }
    } else {
        let (r0, r1, r2) = (rs[0], rs[1], rs[2]);
        let z = 0.0f32;
        let defs: [([f32; 12], usize, &str); 6] = [
            ([z, z, z, z, r1, z, z, r1, z, z, r2, r2], 0, "west"),
            ([r0, r0, r0, r0, z, r1, r1, z, z, z, r2, r2], 1, "east"),
            ([z, r0, r0, z, z, z, z, z, z, z, r2, r2], 2, "south"),
            ([r0, z, z, r0, r1, r1, r1, r1, z, z, r2, r2], 3, "north"),
            ([r0, z, z, r0, z, z, r1, r1, z, z, z, z], 4, "floor"),
            ([r0, r0, z, z, z, r1, r1, z, r2, r2, r2, r2], 5, "ceiling"),
        ];
        for (vals, c, name) in defs.iter() {
            walls.push(WallData::<D>::new(
                cols_from_rowmajor::<D>(vals),
                col(abs, *c),
                col(scat, *c),
                name.to_string(),
            )?);
        }
    }
    Ok(walls)
}

impl<const D: usize> Geometry<D> {
    pub fn init(&mut self) -> Result<(), RoomError> {
        if self.walls.len() > D {
            self.n_bands = self.walls[0].n_bands();
            for w in &self.walls {
                if self.n_bands != w.n_bands() {
                    return Err(RoomError(
                        "Error: All walls should have the same number of frequency bands".into(),
                    ));
                }
            }
        } else if D == 2 {
            return Err(RoomError("Error: The minimum number of walls is 3".into()));
        } else {
            return Err(RoomError("Error: The minimum number of walls is 4".into()));
        }
        self.max_dist = self.get_max_distance();
        Ok(())
    }

    pub fn get_max_distance(&self) -> f32 {
        let (mut max_x, mut max_y, mut max_z) = (0.0f32, 0.0f32, 0.0f32);
        let (mut min_x, mut min_y, mut min_z) = (0.0f32, 0.0f32, 0.0f32);
        for (i, w) in self.walls.iter().enumerate() {
            let mut mx = [0.0f32; 3];
            let mut mn = [0.0f32; 3];
            for d in 0..D {
                let mut a = w.corners[0][d];
                let mut b = w.corners[0][d];
                for c in &w.corners[1..] {
                    if c[d] > a {
                        a = c[d];
                    }
                    if c[d] < b {
                        b = c[d];
                    }
                }
                mx[d] = a;
                mn[d] = b;
            }
            if i == 0 {
                max_x = mx[0];
                min_x = mn[0];
                max_y = mx[1];
                min_y = mn[1];
                max_z = mx[2];
                min_z = mn[2];
            } else {
                if mx[0] > max_x {
                    max_x = mx[0];
                }
                if mx[1] > max_y {
                    max_y = mx[1];
                }
                if mx[2] > max_z {
                    max_z = mx[2];
                }
                if mn[0] < min_x {
                    min_x = mn[0];
                }
                if mn[1] < min_y {
                    min_y = mn[1];
                }
                if mn[2] < min_z {
                    min_z = mn[2];
                }
            }
        }
        let d = [min_x - max_x, min_y - max_y, min_z - max_z];
        norm(&d) + 1.0
    }

    // ------------------------------------------------------------------
    // Image source model
    // ------------------------------------------------------------------

    pub fn image_source_model(
        &self,
        p: &Params,
        source: &[f32; D],
        mics: &[[f32; D]],
        prev: &mut IsmResults<D>,
    ) -> i32 {
        if self.is_shoebox {
            self.image_source_shoebox(p, source, mics, prev)
        } else {
            let mut out: Vec<ImgSrc<D>> = vec![];
            let real = ImgSrc::new_at(*source, self.n_bands, mics.len());
            let mut stack = vec![real];
            self.image_sources_dfs(p, &mut stack, 0, p.ism_order, mics, &mut out);
            let n = out.len();
            if n > 0 {
                let m = mics.len();
                prev.sources = out.iter().map(|s| s.loc).collect();
                prev.gen_walls = out.iter().map(|s| s.gen_wall).collect();
                prev.orders = out.iter().map(|s| s.order).collect();
                prev.attenuations = out.iter().flat_map(|s| s.attenuation.iter().copied()).collect();
                prev.visible_mics = out.iter().flat_map(|s| s.visible.iter().copied()).collect();
                prev.n_sources = n;
                let mut sd = Vec::with_capacity(n * m);
                for s in &out {
                    for k in 0..m {
                        sd.push(s.source_direction[k]);
                    }
                }
                prev.source_directions = sd;
                prev.n_bands = self.n_bands;
                prev.n_mics = m;
            }
            n as i32
        }
    }

    fn image_sources_dfs(
        &self,
        p: &Params,
        stack: &mut Vec<ImgSrc<D>>,
        depth: usize,
        max_order: i32,
        mics: &[[f32; D]],
        out: &mut Vec<ImgSrc<D>>,
    ) {
        // new_is: fresh image source for this call
        let fresh = ImgSrc::new(self.n_bands, mics.len());
        if stack.len() > depth + 1 {
            stack[depth + 1] = fresh;
            stack.truncate(depth + 2);
        } else {
            stack.push(fresh);
        }

        let mut any_visible = false;
        for (m, mic) in mics.iter().enumerate() {
            let mut sd = stack[depth].source_direction[m];
            let vis = self.is_visible_dfs(mic, stack, depth, &mut sd);
            stack[depth].source_direction[m] = sd;
            stack[depth].visible[m] = vis;
            any_visible |= vis;
        }
        if any_visible {
            out.push(stack[depth].clone());
        }
        if max_order == 0 {
            return;
        }
        for wi in 0..self.walls.len() {
            let wall = &self.walls[wi];
            let loc = stack[depth].loc;
            let mut new_loc = stack[depth + 1].loc;
            let dir = wall.reflect(&loc, &mut new_loc);
            stack[depth + 1].loc = new_loc;
            if dir <= 0 {
                continue;
            }
            let mut att: Vec<f32> = stack[depth]
                .attenuation
                .iter()
                .zip(&wall.transmission)
                .map(|(a, t)| a * t)
                .collect();
            if vmax(&wall.scatter) > 0.0 && p.is_hybrid_sim {
                for (a, s) in att.iter_mut().zip(&wall.scatter) {
                    *a *= (1.0f32 - s).sqrt();
                }
            }
            let order = stack[depth].order + 1;
            {
                let ni = &mut stack[depth + 1];
                ni.attenuation = att;
                ni.order = order;
                ni.gen_wall = wi as i32;
            }
            self.image_sources_dfs(p, stack, depth + 1, max_order - 1, mics, out);
        }
    }

    fn is_visible_dfs(
        &self,
        p: &[f32; D],
        chain: &[ImgSrc<D>],
        level: usize,
        source_direction: &mut [f32; D],
    ) -> bool {
        if self.is_obstructed_dfs(p, chain, level) {
            return false;
        }
        let is = &chain[level];
        if level > 0 {
            let wall_id = is.gen_wall as usize;
            let mut inter = [0.0f32; D];
            let ret = self.walls[wall_id].intersection(p, &is.loc, &mut inter);
            if ret >= 0 {
                return self.is_visible_dfs(&inter, chain, level - 1, source_direction);
            } else {
                return false;
            }
        }
        *source_direction = normalized(&sub(p, &is.loc));
        true
    }

    fn is_obstructed_dfs(&self, p: &[f32; D], chain: &[ImgSrc<D>], level: usize) -> bool {
        let is = &chain[level];
        let gen_wall_id = is.gen_wall;
        for &wall_id in &self.obstructing_walls {
            if wall_id != gen_wall_id {
                let mut inter = [0.0f32; D];
                let ret = self.walls[wall_id as usize].intersection(&is.loc, p, &mut inter);
                if ret == WALL_ISECT_VALID || ret == WALL_ISECT_VALID_BNDRY {
                    if level > 0 {
                        let gw = &self.walls[is.gen_wall as usize];
                        let img_side = gw.side(&is.loc);
                        let intersection_side = gw.side(&inter);
                        if img_side != intersection_side && intersection_side != 0 {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn image_source_shoebox(
        &self,
        p: &Params,
        source: &[f32; D],
        mics: &[[f32; D]],
        res: &mut IsmResults<D>,
    ) -> i32 {
        let nb = self.n_bands;
        let nw = 2 * D;
        let order = p.ism_order;
        // transmission_pwr[i][band][wall]
        let mut tp: Vec<Vec<Vec<f32>>> = Vec::with_capacity((order.max(0) + 1) as usize);
        for _ in 0..=order.max(0) {
            tp.push(vec![vec![0.0f32; nw]; nb]);
        }
        for b in 0..nb {
            for w in 0..nw {
                tp[0][b][w] = 1.0;
            }
        }
        if order > 0 {
            let mut scat_max = f32::NEG_INFINITY;
            for row in &self.shoebox_scattering {
                for &v in row {
                    if v > scat_max {
                        scat_max = v;
                    }
                }
            }
            for b in 0..nb {
                for w in 0..nw {
                    tp[1][b][w] = (1.0f32 - self.shoebox_absorption[b][w]).sqrt();
                }
            }
            if scat_max > 0.0 && p.is_hybrid_sim {
                for b in 0..nb {
                    for w in 0..nw {
                        tp[1][b][w] *= (1.0f32 - self.shoebox_scattering[b][w]).sqrt();
                    }
                }
            }
        }
        for i in 2..=(order.max(0) as usize) {
            for b in 0..nb {
                for w in 0..nw {
                    tp[i][b][w] = tp[i - 1][b][w] * tp[1][b][w];
                }
            }
        }

        let o = order as i64;
        let n_image_sources: usize = if D == 2 {
            (1 + 2 * o * (o + 1)) as usize
        } else {
            let osq = o * o;
            (1 + 2 * o * (2 * osq + 3 * o + 4) / 3) as usize
        };
        let m = mics.len();

        // enumerate the lattice points in the original loop order
        let z_max: i32 = if D == 2 { 0 } else { order };
        let mut points: Vec<[i32; 3]> = Vec::with_capacity(n_image_sources);
        for pz in -z_max..=z_max {
            let y_max = order - pz.abs();
            for py in -y_max..=y_max {
                let x_max = (y_max - py.abs()).max(0);
                for px in -x_max..=x_max {
                    points.push([px, py, pz]);
                }
            }
        }
        let n = points.len();

        let mut sources = vec![[0.0f32; D]; n];
        let mut orders = vec![0i32; n];
        let mut orders_xyz = vec![[0i32; D]; n];
        // padded so that chunking never yields fewer chunks than `sources`
        let mut atts = vec![0.0f32; n * nb.max(1)];
        let mut sdirs = vec![[0.0f32; D]; n * m.max(1)];

        // every image source is computed independently: parallel and exact
        use rayon::prelude::*;
        const CH: usize = 1024;
        sources
            .par_chunks_mut(CH)
            .zip(orders.par_chunks_mut(CH))
            .zip(orders_xyz.par_chunks_mut(CH))
            .zip(atts.par_chunks_mut(CH * nb.max(1)))
            .zip(sdirs.par_chunks_mut(CH * m.max(1)))
            .enumerate()
            .for_each(|(ci, ((((src_c, ord_c), oxyz_c), att_c), sd_c))| {
                let mut sd = vec![[0.0f32; D]; m];
                for li in 0..src_c.len() {
                    let point = points[ci * CH + li];
                    let att = &mut att_c[li * nb..(li + 1) * nb];
                    for v in att.iter_mut() {
                        *v = 1.0;
                    }
                    let mut loc = [0.0f32; D];
                    let mut ord = 0i32;
                    for d in 0..D {
                        let abs_point = point[d].abs();
                        let step = if abs_point % 2 == 1 {
                            self.shoebox_size[d] - source[d]
                        } else {
                            source[d]
                        };
                        loc[d] = (point[d] as f32) * self.shoebox_size[d] + step;
                        oxyz_c[li][d] = point[d];
                        for (k, mic) in mics.iter().enumerate() {
                            if abs_point % 2 == 1 {
                                sd[k][d] = loc[d] - mic[d];
                            } else {
                                sd[k][d] = mic[d] - loc[d];
                            }
                        }
                        ord += abs_point;
                        let (mut p1, mut p2) = (0i32, 0i32);
                        if point[d] > 0 {
                            p1 = point[d] / 2;
                            p2 = (point[d] + 1) / 2;
                        } else if point[d] < 0 {
                            p1 = ((point[d] - 1) / 2).abs();
                            p2 = (point[d] / 2).abs();
                        }
                        let (t1, t2) = (&tp[p1 as usize], &tp[p2 as usize]);
                        for b in 0..nb {
                            att[b] *= t1[b][2 * d];
                        }
                        for b in 0..nb {
                            att[b] *= t2[b][2 * d + 1];
                        }
                    }
                    src_c[li] = loc;
                    ord_c[li] = ord;
                    for k in 0..m {
                        sd_c[li * m + k] = normalized(&sd[k]);
                    }
                }
            });

        atts.truncate(n * nb);
        sdirs.truncate(n * m);
        if n_image_sources > 0 {
            res.sources = sources;
            res.orders = orders;
            res.orders_xyz = orders_xyz;
            res.gen_walls = vec![-1; n_image_sources];
            res.attenuations = atts;
            res.visible_mics = vec![true; m * n_image_sources];
            res.source_directions = sdirs;
            res.n_bands = nb;
            res.n_mics = m;
            res.n_sources = n_image_sources;
        }
        n_image_sources as i32
    }

    // ------------------------------------------------------------------
    // Ray tracing
    // ------------------------------------------------------------------

    #[inline]
    pub fn next_wall_hit(&self, start: &[f32; D], end: &[f32; D], scattered_ray: bool) -> ([f32; D], i32, f32) {
        const SHOEBOX_ORDERS: [[usize; 3]; 3] = [[0, 1, 2], [1, 0, 2], [2, 0, 1]];
        let mut result = [0.0f32; D];
        let mut next_wall_index = -1i32;
        let mut hit_dist = self.max_dist;
        let e = eps();

        if self.is_shoebox {
            if scattered_ray {
                return (result, -1, 0.0);
            }
            let dir = sub(end, start);
            'planes: for so in SHOEBOX_ORDERS.iter().take(D) {
                let d = so;
                let abs_dir0 = dir[d[0]].abs();
                if abs_dir0 < e {
                    continue;
                }
                let distance;
                let ind_inc;
                if dir[d[0]] < 0.0 {
                    result[d[0]] = 0.0;
                    distance = start[d[0]];
                    ind_inc = 0;
                } else {
                    result[d[0]] = self.shoebox_size[d[0]];
                    distance = self.shoebox_size[d[0]] - start[d[0]];
                    ind_inc = 1;
                }
                if distance < e {
                    continue;
                }
                let ratio = distance / abs_dir0;
                for i in 1..D {
                    result[d[i]] = start[d[i]] + ratio * dir[d[i]];
                    if result[d[i]] <= -e || self.shoebox_size[d[i]] + e <= result[d[i]] {
                        continue 'planes;
                    }
                }
                next_wall_index = (2 * d[0] + ind_inc) as i32;
                hit_dist = norm(&sub(&result, start));
                break;
            }
        } else {
            let n_walls = if scattered_ray { self.obstructing_walls.len() } else { self.walls.len() };
            for i in 0..n_walls {
                let w = if scattered_ray {
                    &self.walls[self.obstructing_walls[i] as usize]
                } else {
                    &self.walls[i]
                };
                let mut temp_hit = [0.0f32; D];
                let ret = w.intersection(start, end, &mut temp_hit);
                if ret > -1 {
                    let temp_dist = norm(&sub(&temp_hit, start));
                    if temp_dist > e && temp_dist < hit_dist {
                        hit_dist = temp_dist;
                        result = temp_hit;
                        next_wall_index = i as i32;
                    }
                }
            }
        }
        (result, next_wall_index, hit_dist)
    }

    /// Returns true if the scattered ray reached all the microphones.
    #[inline]
    pub fn scat_ray<L: Logger<D>>(
        &self,
        p: &Params,
        mics: &[MicData<D>],
        logger: &mut L,
        transmitted: &[f32],
        wall: &WallData<D>,
        prev_last_hit: &[f32; D],
        hit_point: &[f32; D],
        travel_dist: f32,
        energy_buf: &mut Vec<f32>,
    ) -> bool {
        let distance_thres = p.time_thres * p.sound_speed;
        let mut ret = true;
        for (k, mic) in mics.iter().enumerate() {
            let mic_pos = &mic.loc;
            if wall.side(mic_pos) != wall.side(prev_last_hit) {
                ret = false;
                continue;
            }
            let mut next_wall_index = -1;
            if !self.is_shoebox {
                next_wall_index = self.next_wall_hit(hit_point, mic_pos, true).1;
            }
            if next_wall_index == -1 {
                let hit_point_to_mic = sub(mic_pos, hit_point);
                let hop_dist = norm(&hit_point_to_mic);
                let travel_dist_at_mic = travel_dist + hop_dist;
                let h_sq = (hop_dist * hop_dist) as f64;
                let p_hit_equal =
                    (1.0f64 - (1.0f64 - p.mic_radius_sq / max_d(p.mic_radius_sq, h_sq)).sqrt()) as f32;
                let p_lambert = 2.0f32 * wall.cosine_angle(&hit_point_to_mic).abs();
                energy_buf.clear();
                let mut mx = f32::NEG_INFINITY;
                for (s, t) in wall.scatter.iter().zip(transmitted) {
                    let v = ((s * t) * p_hit_equal) * p_lambert;
                    if v > mx {
                        mx = v;
                    }
                    energy_buf.push(v);
                }
                if travel_dist_at_mic < distance_thres && mx > p.energy_thres {
                    let r_sq = (travel_dist_at_mic as f64) * (travel_dist_at_mic as f64);
                    let p_hit = 1.0f64 - (1.0f64 - p.mic_radius_sq / max_d(p.mic_radius_sq, r_sq)).sqrt();
                    let div = (r_sq * p_hit) as f32;
                    for v in energy_buf.iter_mut() {
                        *v = *v / div;
                    }
                    logger.log(k, mic, travel_dist_at_mic, energy_buf, hit_point);
                } else {
                    ret = false;
                }
            } else {
                ret = false;
            }
        }
        ret
    }

    pub fn simul_ray<L: Logger<D>>(
        &self,
        p: &Params,
        mics: &[MicData<D>],
        logger: &mut L,
        rng: &mut Mt64,
        ray_direction: &[f32; D],
        source_pos: &[f32; D],
        energy_0: &[f32],
    ) {
        let n_bands = self.n_bands;
        let mut start = *source_pos;
        let mut dir = *ray_direction;
        let mut transmitted: Vec<f32> = (0..n_bands).map(|b| 1.0f32 * energy_0[b]).collect();
        let mut energy_buf: Vec<f32> = Vec::with_capacity(n_bands);
        let mut travel_dist = 0.0f32;
        let mut specular_counter = 0i32;
        let e_thres = vmax(energy_0) * p.energy_thres;
        let distance_thres = p.time_thres * p.sound_speed;
        let e = eps();

        loop {
            let mut end = [0.0f32; D];
            for i in 0..D {
                end[i] = start[i] + dir[i] * self.max_dist;
            }
            let (hit_point, next_wall_index, hit_distance) = self.next_wall_hit(&start, &end, false);
            if next_wall_index == -1 {
                break;
            }
            let wall = &self.walls[next_wall_index as usize];

            if !(p.is_hybrid_sim && specular_counter < p.ism_order) {
                for (k, mic) in mics.iter().enumerate() {
                    let to_mic = sub(&mic.loc, &start);
                    let impact_distance = dot(&to_mic, &dir);
                    let impacts = -e < impact_distance && impact_distance < hit_distance + e;
                    if impacts {
                        let mut v = [0.0f32; D];
                        for i in 0..D {
                            v[i] = to_mic[i] - dir[i] * impact_distance;
                        }
                        if norm(&v) < p.mic_radius + e {
                            let distance = impact_distance.abs();
                            let travel_dist_at_mic = travel_dist + distance;
                            let r_sq = (travel_dist_at_mic as f64) * (travel_dist_at_mic as f64);
                            let p_hit =
                                1.0f64 - (1.0f64 - p.mic_radius_sq / max_d(p.mic_radius_sq, r_sq)).sqrt();
                            let div = (r_sq * p_hit) as f32;
                            energy_buf.clear();
                            energy_buf.extend(transmitted.iter().map(|t| t / div));
                            logger.log(k, mic, travel_dist_at_mic, &energy_buf, &start);
                        }
                    }
                }
            }

            travel_dist += hit_distance;
            for (t, r) in transmitted.iter_mut().zip(&wall.energy_reflection) {
                *t *= r;
            }

            if wall.does_scatter {
                self.scat_ray(
                    p,
                    mics,
                    logger,
                    &transmitted,
                    wall,
                    &start,
                    &hit_point,
                    travel_dist,
                    &mut energy_buf,
                );
            }

            if travel_dist > distance_thres || vmax(&transmitted) < e_thres {
                break;
            }

            specular_counter += 1;
            dir = wall.normal_reflect_dir(&dir);
            if wall.does_scatter {
                let scat_dir = neg(&wall.sample_lambertian_reflection(rng));
                let a = wall.average_scatter;
                for i in 0..D {
                    dir[i] = a * scat_dir[i] + (1.0f32 - a) * dir[i];
                }
                dir = normalized(&dir);
            }
            start = hit_point;
        }
    }

    pub fn contains(&self, point: &[f32; D], rng: &mut Mt64) -> bool {
        let n_walls = self.walls.len();
        let mut min_coord = [0.0f32; D];
        for (i, w) in self.walls.iter().enumerate() {
            let mut mn = w.corners[0];
            for c in &w.corners[1..] {
                for d in 0..D {
                    if c[d] < mn[d] {
                        mn[d] = c[d];
                    }
                }
            }
            if i == 0 {
                min_coord = mn;
            } else {
                for d in 0..D {
                    if mn[d] < min_coord[d] {
                        min_coord[d] = mn[d];
                    }
                }
            }
        }
        let mut outside_point = min_coord;
        let mut n_intersections;
        loop {
            n_intersections = 0usize;
            let mut ambiguous = false;
            outside_point[0] -= rng.uniform(0.05f32, 0.54f32);
            outside_point[1] -= rng.uniform(0.05f32, 0.8461f32);
            if D == 3 {
                outside_point[2] -= rng.uniform(0.05f32, 0.501f32);
            }
            for i in 0..n_walls {
                let result = self.walls[i].intersects(&outside_point, point);
                ambiguous = ambiguous || (result > 0);
                if result > -1 {
                    n_intersections += 1;
                }
            }
            if !ambiguous {
                break;
            }
        }
        n_intersections % 2 == 1
    }

    pub fn any_wall_scatters(&self) -> bool {
        self.walls.iter().any(|w| w.does_scatter)
    }
}

#[inline(always)]
fn max_d(a: f64, b: f64) -> f64 {
    // std::max(a, b)
    if a < b {
        b
    } else {
        a
    }
}

/// Sink for histogram entries produced by the ray tracer.
pub trait Logger<const D: usize> {
    fn log(&mut self, mic_idx: usize, mic: &MicData<D>, distance: f32, energy: &[f32], origin: &[f32; D]);
}

/// Event recorded for later (ordered) application into the histograms.
#[derive(Clone, Copy)]
pub struct Event {
    pub mic: u32,
    pub dir: u32,
    pub col: u32,
}

#[derive(Default)]
pub struct EventLog {
    pub events: Vec<Event>,
    pub energies: Vec<f32>,
}

impl<const D: usize> Logger<D> for EventLog {
    #[inline]
    fn log(&mut self, mic_idx: usize, mic: &MicData<D>, distance: f32, energy: &[f32], origin: &[f32; D]) {
        let col = mic.dist_bin(distance) as u32;
        let dir = mic.get_dir_bin(origin) as u32;
        self.events.push(Event { mic: mic_idx as u32, dir, col });
        self.energies.extend_from_slice(energy);
    }
}

impl EventLog {
    pub fn clear(&mut self) {
        self.events.clear();
        self.energies.clear();
    }
}

#[derive(Clone)]
pub struct ImgSrc<const D: usize> {
    pub loc: [f32; D],
    pub attenuation: Vec<f32>,
    pub order: i32,
    pub gen_wall: i32,
    pub visible: Vec<bool>,
    pub source_direction: Vec<[f32; D]>,
}

impl<const D: usize> ImgSrc<D> {
    fn new(n_bands: usize, n_mics: usize) -> Self {
        ImgSrc {
            loc: [0.0; D],
            attenuation: vec![1.0; n_bands],
            order: 0,
            gen_wall: -1,
            visible: vec![false; n_mics],
            source_direction: vec![[0.0; D]; n_mics],
        }
    }
    fn new_at(loc: [f32; D], n_bands: usize, n_mics: usize) -> Self {
        let mut s = Self::new(n_bands, n_mics);
        s.loc = loc;
        s
    }
}

// ---------------------------------------------------------------------------
// Split ray tracing: sequential trajectory (RNG dependent) + parallel logging
// ---------------------------------------------------------------------------

/// One bounce of a ray: everything `simul_ray` needs to log the
/// specular and scattered contributions of that bounce.
#[derive(Clone, Copy)]
pub struct Bounce<const D: usize> {
    pub start: [f32; D],
    pub dir: [f32; D],
    pub hit_point: [f32; D],
    pub wall: u32,
    pub hit_distance: f32,
    pub travel_before: f32,
    pub travel_after: f32,
    pub specular_active: bool,
}

/// Trajectories of a batch of rays.
#[derive(Default)]
pub struct Paths<const D: usize> {
    pub bounces: Vec<Bounce<D>>,
    /// transmitted energy after each bounce (n_bands per bounce)
    pub trans_after: Vec<f32>,
    /// transmitted energy at the start of each ray (n_bands per ray)
    pub trans_init: Vec<f32>,
    /// ray r owns bounces[ray_start[r]..ray_start[r+1]]
    pub ray_start: Vec<usize>,
}

impl<const D: usize> Paths<D> {
    pub fn clear(&mut self) {
        self.bounces.clear();
        self.trans_after.clear();
        self.trans_init.clear();
        self.ray_start.clear();
        self.ray_start.push(0);
    }
    pub fn n_rays(&self) -> usize {
        self.ray_start.len() - 1
    }
}

impl<const D: usize> Geometry<D> {
    /// The geometric (and RNG consuming) part of `simul_ray`. Must perform the
    /// same operations as `simul_ray`.
    pub fn trace_path(
        &self,
        p: &Params,
        rng: &mut Mt64,
        ray_direction: &[f32; D],
        source_pos: &[f32; D],
        energy_0: &[f32],
        paths: &mut Paths<D>,
    ) {
        let n_bands = self.n_bands;
        let mut start = *source_pos;
        let mut dir = *ray_direction;
        let t0 = paths.trans_after.len();
        let _ = t0;
        let mut transmitted: Vec<f32> = (0..n_bands).map(|b| 1.0f32 * energy_0[b]).collect();
        paths.trans_init.extend_from_slice(&transmitted);
        let mut travel_dist = 0.0f32;
        let mut specular_counter = 0i32;
        let e_thres = vmax(energy_0) * p.energy_thres;
        let distance_thres = p.time_thres * p.sound_speed;

        loop {
            let mut end = [0.0f32; D];
            for i in 0..D {
                end[i] = start[i] + dir[i] * self.max_dist;
            }
            let (hit_point, next_wall_index, hit_distance) = self.next_wall_hit(&start, &end, false);
            if next_wall_index == -1 {
                break;
            }
            let wall = &self.walls[next_wall_index as usize];
            let specular_active = !(p.is_hybrid_sim && specular_counter < p.ism_order);
            let travel_before = travel_dist;
            travel_dist += hit_distance;
            for (t, r) in transmitted.iter_mut().zip(&wall.energy_reflection) {
                *t *= r;
            }
            paths.bounces.push(Bounce {
                start,
                dir,
                hit_point,
                wall: next_wall_index as u32,
                hit_distance,
                travel_before,
                travel_after: travel_dist,
                specular_active,
            });
            paths.trans_after.extend_from_slice(&transmitted);

            if travel_dist > distance_thres || vmax(&transmitted) < e_thres {
                break;
            }
            specular_counter += 1;
            dir = wall.normal_reflect_dir(&dir);
            if wall.does_scatter {
                let scat_dir = neg(&wall.sample_lambertian_reflection(rng));
                let a = wall.average_scatter;
                for i in 0..D {
                    dir[i] = a * scat_dir[i] + (1.0f32 - a) * dir[i];
                }
                dir = normalized(&dir);
            }
            start = hit_point;
        }
        paths.ray_start.push(paths.bounces.len());
    }

    /// The logging part of `simul_ray` for ray `r` of `paths`, emitting events
    /// in exactly the same order as `simul_ray`.
    pub fn emit_path<L: Logger<D>>(&self, p: &Params, mics: &[MicData<D>], paths: &Paths<D>, r: usize, logger: &mut L) {
        let nb = self.n_bands;
        let e = eps();
        let mut energy_buf: Vec<f32> = Vec::with_capacity(nb);
        for bi in paths.ray_start[r]..paths.ray_start[r + 1] {
            let b = &paths.bounces[bi];
            let trans_before: &[f32] = if bi == paths.ray_start[r] {
                &paths.trans_init[r * nb..(r + 1) * nb]
            } else {
                &paths.trans_after[(bi - 1) * nb..bi * nb]
            };
            let trans_after = &paths.trans_after[bi * nb..(bi + 1) * nb];
            if b.specular_active {
                for (k, mic) in mics.iter().enumerate() {
                    let to_mic = sub(&mic.loc, &b.start);
                    let impact_distance = dot(&to_mic, &b.dir);
                    let impacts = -e < impact_distance && impact_distance < b.hit_distance + e;
                    if impacts {
                        let mut v = [0.0f32; D];
                        for i in 0..D {
                            v[i] = to_mic[i] - b.dir[i] * impact_distance;
                        }
                        if norm(&v) < p.mic_radius + e {
                            let distance = impact_distance.abs();
                            let travel_dist_at_mic = b.travel_before + distance;
                            let r_sq = (travel_dist_at_mic as f64) * (travel_dist_at_mic as f64);
                            let p_hit =
                                1.0f64 - (1.0f64 - p.mic_radius_sq / max_d(p.mic_radius_sq, r_sq)).sqrt();
                            let div = (r_sq * p_hit) as f32;
                            energy_buf.clear();
                            energy_buf.extend(trans_before.iter().map(|t| t / div));
                            logger.log(k, mic, travel_dist_at_mic, &energy_buf, &b.start);
                        }
                    }
                }
            }
            let wall = &self.walls[b.wall as usize];
            if wall.does_scatter {
                self.scat_ray(
                    p,
                    mics,
                    logger,
                    trans_after,
                    wall,
                    &b.start,
                    &b.hit_point,
                    b.travel_after,
                    &mut energy_buf,
                );
            }
        }
    }
}
