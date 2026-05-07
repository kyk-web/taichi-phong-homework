import taichi as ti
import taichi.math as tm

INF = 1e8
EPS = 1e-4

# -----------------------------
# Scene geometry constants
# -----------------------------
SPHERE_CENTER = tm.vec3(-1.2, -0.2, 0.0)
SPHERE_RADIUS = 1.2

# 注意：
# 你当前 app 里的屏幕 y 方向和数学坐标显示方向是反着的，
# 所以为了让“视觉上”圆锥是正常直立的，这里把 apex/base 的 y 反过来设。
CONE_APEX = tm.vec3(1.2, -1.4, 0.0)   # 视觉上显示在上方的尖点
CONE_BASE_Y = 1.2                     # 视觉上显示在下方的底面
CONE_RADIUS = 1.2
CONE_HEIGHT = CONE_BASE_Y - CONE_APEX.y

# Cone local equation:
# apex at origin, axis toward +y
# x^2 + z^2 = (k * y)^2, y in [0, h]
CONE_K = CONE_RADIUS / CONE_HEIGHT
CONE_K2 = CONE_K * CONE_K


@ti.func
def safe_normalize(v):
    n2 = tm.dot(v, v)
    out = tm.vec3(0.0, 0.0, 0.0)
    if n2 > 1e-12:
        out = v / ti.sqrt(n2)
    return out


@ti.func
def point_at(ray_o, ray_d, t):
    return ray_o + t * ray_d


@ti.func
def sphere_intersection(ray_o, ray_d):
    t_hit = INF
    n_hit = tm.vec3(0.0, 0.0, 0.0)

    oc = ray_o - SPHERE_CENTER

    a = tm.dot(ray_d, ray_d)
    b = 2.0 * tm.dot(oc, ray_d)
    c = tm.dot(oc, oc) - SPHERE_RADIUS * SPHERE_RADIUS

    disc = b * b - 4.0 * a * c

    if disc >= 0.0:
        sqrt_disc = ti.sqrt(disc)
        inv_2a = 0.5 / a

        t0 = (-b - sqrt_disc) * inv_2a
        t1 = (-b + sqrt_disc) * inv_2a

        if t0 > EPS:
            t_hit = t0
        elif t1 > EPS:
            t_hit = t1

        if t_hit < INF:
            p = point_at(ray_o, ray_d, t_hit)
            n_hit = safe_normalize(p - SPHERE_CENTER)

    return t_hit, n_hit


@ti.func
def cone_intersection(ray_o, ray_d):
    """
    这版使用“视觉上直立”的圆锥：
    apex   = (1.2, -1.4, 0)
    base y = 1.2
    radius = 1.2

    Returns:
        (t_hit, normal_hit)
    If no hit:
        (INF, zero_vec3)
    """
    hit_t = INF
    hit_n = tm.vec3(0.0, 0.0, 0.0)

    # Transform to cone local coordinates: apex as origin
    o = ray_o - CONE_APEX
    d = ray_d

    # ---------------------------------
    # 1) Side surface intersection
    # Infinite cone equation:
    # x^2 + z^2 - k^2 y^2 = 0
    # valid only when y in [0, h]
    # ---------------------------------
    t_side = INF
    n_side = tm.vec3(0.0, 0.0, 0.0)
    valid_side = 0

    a = d.x * d.x + d.z * d.z - CONE_K2 * d.y * d.y
    b = 2.0 * (o.x * d.x + o.z * d.z - CONE_K2 * o.y * d.y)
    c = o.x * o.x + o.z * o.z - CONE_K2 * o.y * o.y

    disc = b * b - 4.0 * a * c

    if ti.abs(a) > 1e-8 and disc >= 0.0:
        sqrt_disc = ti.sqrt(disc)
        inv_2a = 0.5 / a

        t0 = (-b - sqrt_disc) * inv_2a
        t1 = (-b + sqrt_disc) * inv_2a

        cand_t = INF

        if t0 > EPS:
            y0 = o.y + t0 * d.y
            if y0 >= 0.0 and y0 <= CONE_HEIGHT:
                cand_t = t0

        if t1 > EPS:
            y1 = o.y + t1 * d.y
            if y1 >= 0.0 and y1 <= CONE_HEIGHT:
                if t1 < cand_t:
                    cand_t = t1

        if cand_t < INF:
            p_local = o + cand_t * d
            # Gradient of F = x^2 + z^2 - k^2 y^2
            n_local = tm.vec3(
                p_local.x,
                -CONE_K2 * p_local.y,
                p_local.z
            )
            t_side = cand_t
            n_side = safe_normalize(n_local)
            valid_side = 1

    # ---------------------------------
    # 2) Base disk intersection
    # Plane: y = h
    # ---------------------------------
    t_base = INF
    n_base = tm.vec3(0.0, 1.0, 0.0)
    valid_base = 0

    if ti.abs(d.y) > 1e-8:
        t_plane = (CONE_HEIGHT - o.y) / d.y
        if t_plane > EPS:
            p_local = o + t_plane * d
            r2 = p_local.x * p_local.x + p_local.z * p_local.z
            if r2 <= CONE_RADIUS * CONE_RADIUS:
                t_base = t_plane
                valid_base = 1

    # ---------------------------------
    # 3) Choose nearest valid hit
    # ---------------------------------
    if valid_side == 1:
        hit_t = t_side
        hit_n = n_side

    if valid_base == 1 and t_base < hit_t:
        hit_t = t_base
        hit_n = n_base

    return hit_t, hit_n