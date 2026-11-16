"""Triton kernel of the GPU ray tracer (see ``cuda_rt``)."""
import triton
import triton.language as tl


@triton.jit
def _rt_kernel(
    ray_dir_ptr,  # (R, 3) f32
    ray_e_ptr,  # (R, NBP) f32, energies (already scaled by ENERGY_0 / n_rays)
    block_job_ptr,  # (n_programs,) i32
    block_ray0_ptr,  # (n_programs,) i32: first global ray of the program
    job_i_ptr,  # (J, 8) i32: ray_end, mic_start, n_mics, hist_start, n_bins, ism_order, hybrid, seed
    job_f_ptr,  # (J, 16) f32: L(3), src(3), e_thres, dist_thres, mic_r, hist_res, energy_thres
    wall_ptr,  # (J, 6, 2 * NBP + 2) f32: refl[NBP], scat[NBP], avg_scat, does_scatter
    mic_ptr,  # (M_total, 3) f32
    hist_ptr,  # f32, per job (n_mics, NB, n_bins)
    eps,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_BOUNCES: tl.constexpr,
):
    pid = tl.program_id(0)
    job = tl.load(block_job_ptr + pid)
    r0 = tl.load(block_ray0_ptr + pid)
    ray_end = tl.load(job_i_ptr + job * 8 + 0)
    mic_start = tl.load(job_i_ptr + job * 8 + 1)
    n_mics = tl.load(job_i_ptr + job * 8 + 2)
    hist_start = tl.load(job_i_ptr + job * 8 + 3).to(tl.int64)
    n_bins = tl.load(job_i_ptr + job * 8 + 4)
    ism_order = tl.load(job_i_ptr + job * 8 + 5)
    hybrid = tl.load(job_i_ptr + job * 8 + 6)
    seed = tl.load(job_i_ptr + job * 8 + 7)

    Lx = tl.load(job_f_ptr + job * 16 + 0)
    Ly = tl.load(job_f_ptr + job * 16 + 1)
    Lz = tl.load(job_f_ptr + job * 16 + 2)
    e_thres = tl.load(job_f_ptr + job * 16 + 6)
    dist_thres = tl.load(job_f_ptr + job * 16 + 7)
    mic_r = tl.load(job_f_ptr + job * 16 + 8)
    hist_res = tl.load(job_f_ptr + job * 16 + 9)
    energy_thres = tl.load(job_f_ptr + job * 16 + 10)
    mic_r2 = mic_r * mic_r

    rays = r0 + tl.arange(0, BLOCK)
    alive = rays < ray_end
    b = tl.arange(0, NBP)
    bmask = b < NB

    dx = tl.load(ray_dir_ptr + rays * 3 + 0, mask=alive, other=1.0)
    dy = tl.load(ray_dir_ptr + rays * 3 + 1, mask=alive, other=0.0)
    dz = tl.load(ray_dir_ptr + rays * 3 + 2, mask=alive, other=0.0)
    sx = tl.zeros([BLOCK], tl.float32) + tl.load(job_f_ptr + job * 16 + 3)
    sy = tl.zeros([BLOCK], tl.float32) + tl.load(job_f_ptr + job * 16 + 4)
    sz = tl.zeros([BLOCK], tl.float32) + tl.load(job_f_ptr + job * 16 + 5)
    trans = tl.load(
        ray_e_ptr + rays[:, None] * NBP + b[None, :],
        mask=alive[:, None] & bmask[None, :],
        other=0.0,
    )
    travel = tl.zeros([BLOCK], tl.float32)
    spec = tl.zeros([BLOCK], tl.int32)
    inf = float("inf")
    WROW: tl.constexpr = 2 * NBP + 2

    bounce = 0
    while (tl.max(alive.to(tl.int32), axis=0) > 0) & (bounce < MAX_BOUNCES):
        # ---- next wall hit (shoebox: nearest plane along the ray)
        distx = tl.where(dx < 0, sx, Lx - sx)
        disty = tl.where(dy < 0, sy, Ly - sy)
        distz = tl.where(dz < 0, sz, Lz - sz)
        tx = tl.where((tl.abs(dx) < eps) | (distx < eps), inf, distx / tl.abs(dx))
        ty = tl.where((tl.abs(dy) < eps) | (disty < eps), inf, disty / tl.abs(dy))
        tz = tl.where((tl.abs(dz) < eps) | (distz < eps), inf, distz / tl.abs(dz))
        t = tl.minimum(tx, tl.minimum(ty, tz))
        alive = alive & (t < inf)
        ax = tl.where(tx == t, 0, tl.where(ty == t, 1, 2))
        hx = sx + t * dx
        hy = sy + t * dy
        hz = sz + t * dz
        hx = tl.where(ax == 0, tl.where(dx < 0, 0.0, Lx), hx)
        hy = tl.where(ax == 1, tl.where(dy < 0, 0.0, Ly), hy)
        hz = tl.where(ax == 2, tl.where(dz < 0, 0.0, Lz), hz)
        hit_dist = tl.sqrt((hx - sx) * (hx - sx) + (hy - sy) * (hy - sy) + (hz - sz) * (hz - sz))
        dax = tl.where(ax == 0, dx, tl.where(ax == 1, dy, dz))
        w = 2 * ax + (dax > 0).to(tl.int32)
        nsign = tl.where(dax > 0, 1.0, -1.0)  # outward normal along axis ax

        # ---- specular contributions to the receivers
        log_spec = alive & ((hybrid == 0) | (spec >= ism_order))
        for k in range(0, n_mics):
            mx = tl.load(mic_ptr + (mic_start + k) * 3 + 0)
            my = tl.load(mic_ptr + (mic_start + k) * 3 + 1)
            mz = tl.load(mic_ptr + (mic_start + k) * 3 + 2)
            tmx = mx - sx
            tmy = my - sy
            tmz = mz - sz
            imp = tmx * dx + tmy * dy + tmz * dz
            vx = tmx - dx * imp
            vy = tmy - dy * imp
            vz = tmz - dz * imp
            hit = log_spec & (imp > -eps) & (imp < hit_dist + eps)
            hit = hit & (tl.sqrt(vx * vx + vy * vy + vz * vz) < mic_r + eps)
            dmic = travel + tl.abs(imp)
            r2 = dmic * dmic
            x = mic_r2 / tl.maximum(mic_r2, r2)
            div = r2 * (x / (1.0 + tl.sqrt(1.0 - x)))  # r^2 * (1 - sqrt(1 - x)), stable
            bin = (dmic / hist_res).to(tl.int32)
            hit = hit & (bin < n_bins)
            off = hist_start + ((k * NB + b[None, :]) * n_bins + bin[:, None]).to(tl.int64)
            tl.atomic_add(hist_ptr + off, trans / div[:, None], mask=hit[:, None] & bmask[None, :])

        travel = travel + hit_dist
        wbase = wall_ptr + job * 6 * WROW + w * WROW
        refl = tl.load(wbase[:, None] + b[None, :], mask=alive[:, None] & bmask[None, :], other=1.0)
        trans = trans * refl
        avg_scat = tl.load(wbase + 2 * NBP, mask=alive, other=0.0)
        does_scat = tl.load(wbase + 2 * NBP + 1, mask=alive, other=0.0) > 0.5

        # ---- diffuse rain from the scattering wall to every receiver
        scat_rows = alive & does_scat
        scat = tl.load(
            wbase[:, None] + NBP + b[None, :],
            mask=scat_rows[:, None] & bmask[None, :],
            other=0.0,
        )
        Lax = tl.where(ax == 0, Lx, tl.where(ax == 1, Ly, Lz))
        sax = tl.where(ax == 0, sx, tl.where(ax == 1, sy, sz))
        # side of the previous point w.r.t. the wall plane
        ip_s = tl.where(dax > 0, sax - Lax, -sax)
        side_s = tl.where(ip_s > eps, 1, tl.where(ip_s < -eps, -1, 0))
        for k in range(0, n_mics):
            mx = tl.load(mic_ptr + (mic_start + k) * 3 + 0)
            my = tl.load(mic_ptr + (mic_start + k) * 3 + 1)
            mz = tl.load(mic_ptr + (mic_start + k) * 3 + 2)
            max_ = tl.where(ax == 0, mx, tl.where(ax == 1, my, mz))
            ip_m = tl.where(dax > 0, max_ - Lax, -max_)
            side_m = tl.where(ip_m > eps, 1, tl.where(ip_m < -eps, -1, 0))
            hmx = mx - hx
            hmy = my - hy
            hmz = mz - hz
            hop = tl.sqrt(hmx * hmx + hmy * hmy + hmz * hmz)
            h2 = hop * hop
            xh = mic_r2 / tl.maximum(mic_r2, h2)
            p_eq = xh / (1.0 + tl.sqrt(1.0 - xh))
            hm_ax = tl.where(ax == 0, hmx, tl.where(ax == 1, hmy, hmz))
            p_lam = 2.0 * tl.abs(hm_ax) / hop
            e = (scat * trans) * (p_eq * p_lam)[:, None]
            emax = tl.max(tl.where(bmask[None, :], e, -inf), axis=1)
            dmic = travel + hop
            ok = scat_rows & (side_m == side_s) & (dmic < dist_thres) & (emax > energy_thres)
            r2 = dmic * dmic
            x = mic_r2 / tl.maximum(mic_r2, r2)
            div = r2 * (x / (1.0 + tl.sqrt(1.0 - x)))
            bin = (dmic / hist_res).to(tl.int32)
            ok = ok & (bin < n_bins)
            off = hist_start + ((k * NB + b[None, :]) * n_bins + bin[:, None]).to(tl.int64)
            tl.atomic_add(hist_ptr + off, e / div[:, None], mask=ok[:, None] & bmask[None, :])

        # ---- termination
        tmax = tl.max(tl.where(bmask[None, :], trans, -inf), axis=1)
        alive = alive & (travel <= dist_thres) & (tmax >= e_thres)

        # ---- new direction: specular, mixed with a Lambertian sample
        spec = spec + 1
        dx = tl.where(ax == 0, -dx, dx)
        dy = tl.where(ax == 1, -dy, dy)
        dz = tl.where(ax == 2, -dz, dz)
        u1 = tl.rand(seed + bounce, rays * 2)
        u2 = tl.rand(seed + bounce, rays * 2 + 1)
        rr = tl.sqrt(u1)
        phi = 6.283185307179586 * u2
        xl = rr * tl.cos(phi)
        yl = rr * tl.sin(phi)
        zl = tl.sqrt(tl.maximum(1.0 - u1, 0.0))
        # scattered direction = -(b0 * xl + b1 * yl + zl * n), b0/b1 the other axes
        snx = tl.where(ax == 0, -zl * nsign, tl.where(ax == 1, -yl, -xl))
        sny = tl.where(ax == 1, -zl * nsign, tl.where(ax == 2, -yl, -xl))
        snz = tl.where(ax == 2, -zl * nsign, tl.where(ax == 0, -yl, -xl))
        a = tl.where(does_scat, avg_scat, 0.0)
        nx = a * snx + (1.0 - a) * dx
        ny = a * sny + (1.0 - a) * dy
        nz = a * snz + (1.0 - a) * dz
        nn = tl.sqrt(nx * nx + ny * ny + nz * nz)
        dx = tl.where(does_scat, nx / nn, dx)
        dy = tl.where(does_scat, ny / nn, dy)
        dz = tl.where(does_scat, nz / nn, dz)
        sx = hx
        sy = hy
        sz = hz
        bounce += 1
