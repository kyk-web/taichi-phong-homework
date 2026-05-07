import taichi as ti

from src.config import CAMERA_POS, LIGHT_COLOR, LIGHT_POS
from src.math_utils import clamp_color, safe_normalize


CAMERA_POS_V = ti.Vector(CAMERA_POS)
LIGHT_POS_V = ti.Vector(LIGHT_POS)
LIGHT_COLOR_V = ti.Vector(LIGHT_COLOR)


@ti.func
def reflect_vec(i, n):
    return i - 2.0 * i.dot(n) * n


@ti.func
def phong_shade(hit_pos, normal, object_color, ka, kd, ks, shininess):
    l = safe_normalize(LIGHT_POS_V - hit_pos)
    v = safe_normalize(CAMERA_POS_V - hit_pos)
    r = safe_normalize(reflect_vec(-l, normal))

    ambient = ka * LIGHT_COLOR_V * object_color
    diffuse = kd * ti.max(0.0, normal.dot(l)) * LIGHT_COLOR_V * object_color
    specular_strength = ks * ti.pow(ti.max(0.0, r.dot(v)), shininess)
    specular = specular_strength * LIGHT_COLOR_V

    return clamp_color(ambient + diffuse + specular)