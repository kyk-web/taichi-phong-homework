import taichi as ti

from src.config import (
    ASPECT,
    BACKGROUND_COLOR,
    CAMERA_POS,
    DEFAULT_KA,
    DEFAULT_KD,
    DEFAULT_KS,
    DEFAULT_SHININESS,
    HEIGHT,
    INF,
    SPHERE_COLOR,
    CONE_COLOR,
    TITLE,
    VIEWPORT_SCALE,
    WIDTH,
)
from src.geometry import cone_intersection, sphere_intersection
from src.math_utils import safe_normalize
from src.shading import phong_shade


BACKGROUND_COLOR_V = ti.Vector(BACKGROUND_COLOR)
CAMERA_POS_V = ti.Vector(CAMERA_POS)
SPHERE_COLOR_V = ti.Vector(SPHERE_COLOR)
CONE_COLOR_V = ti.Vector(CONE_COLOR)


@ti.data_oriented
class PhongDemoApp:
    def __init__(self):
        ti.init(arch=ti.gpu, default_fp=ti.f32)

        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

        self.ka = ti.field(dtype=ti.f32, shape=())
        self.kd = ti.field(dtype=ti.f32, shape=())
        self.ks = ti.field(dtype=ti.f32, shape=())
        self.shininess = ti.field(dtype=ti.f32, shape=())

        self.ka[None] = DEFAULT_KA
        self.kd[None] = DEFAULT_KD
        self.ks[None] = DEFAULT_KS
        self.shininess[None] = DEFAULT_SHININESS

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            u = (2.0 * (i + 0.5) / WIDTH - 1.0) * ASPECT * VIEWPORT_SCALE
            v = (1.0 - 2.0 * (j + 0.5) / HEIGHT) * VIEWPORT_SCALE

            ray_o = CAMERA_POS_V
            ray_d = safe_normalize(ti.Vector([u, v, -1.0]))

            min_t = INF
            hit_normal = ti.Vector([0.0, 0.0, 0.0])
            hit_color = BACKGROUND_COLOR_V
            hit_anything = False

            t_sphere, n_sphere = sphere_intersection(ray_o, ray_d)
            if t_sphere < min_t:
                min_t = t_sphere
                hit_normal = n_sphere
                hit_color = SPHERE_COLOR_V
                hit_anything = True

            t_cone, n_cone = cone_intersection(ray_o, ray_d)
            if t_cone < min_t:
                min_t = t_cone
                hit_normal = n_cone
                hit_color = CONE_COLOR_V
                hit_anything = True

            if hit_anything and min_t < INF * 0.5:
                hit_pos = ray_o + min_t * ray_d
                self.pixels[i, j] = phong_shade(
                    hit_pos,
                    hit_normal,
                    hit_color,
                    self.ka[None],
                    self.kd[None],
                    self.ks[None],
                    self.shininess[None],
                )
            else:
                self.pixels[i, j] = BACKGROUND_COLOR_V

    def run(self):
        window = ti.ui.Window(TITLE, (WIDTH, HEIGHT), vsync=True)
        canvas = window.get_canvas()
        gui = window.get_gui()

        while window.running:
            with gui.sub_window("Material Parameters", 0.58, 0.05, 0.34, 0.25) as w:
                self.ka[None] = w.slider_float("Ka (Ambient)", self.ka[None], 0.0, 1.0)
                self.kd[None] = w.slider_float("Kd (Diffuse)", self.kd[None], 0.0, 1.0)
                self.ks[None] = w.slider_float("Ks (Specular)", self.ks[None], 0.0, 1.0)
                self.shininess[None] = w.slider_float("N (Shininess)", self.shininess[None], 1.0, 128.0)

            self.render()
            canvas.set_image(self.pixels)
            window.show()