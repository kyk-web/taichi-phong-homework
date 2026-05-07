import taichi as ti
from src.config import EPS


@ti.func
def safe_normalize(v):
    norm = ti.sqrt(v.dot(v))
    return v / ti.max(norm, EPS)


@ti.func
def clamp01(x):
    return ti.min(1.0, ti.max(0.0, x))


@ti.func
def clamp_color(c):
    return ti.Vector([
        clamp01(c[0]),
        clamp01(c[1]),
        clamp01(c[2]),
    ])