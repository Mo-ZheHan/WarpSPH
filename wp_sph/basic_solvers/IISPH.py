###########################################################################
# Smoothed Particle Hydrodynamics Implementation
#
# Modified from the original Warp example based on WCSPH
# https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sph.py
#
# Reference Publication
# M. Ihmsen, J. Cornelis, B. Solenthaler, C. Horvath, and M. Teschner.
# "Implicit Incompressible SPH."
# IEEE Trans. Visual. Comput. Graphics. Vol. 20. 2014.
#
###########################################################################

import basic_solvers
import numpy as np
import warp as wp
import warp.render

from ..main import *
from .sph_funcs import *  # TODO cache the functions


@wp.kernel
def count_neighbor(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_pointer: wp.array(dtype=wp.uint32),
    neighbor_num: wp.array(dtype=wp.uint32),
    neighbor_list_index: wp.array(dtype=wp.uint32),
):
    tid = wp.tid()
    neighbor_id = 0

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # count neighbors
    for index in wp.hash_grid_query(grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - particle_x[index])
        if distance < SMOOTHING_LENGTH:
            neighbor_id += 1

    # store number of neighbors
    neighbor_num[i] = neighbor_id
    neighbor_list_index[i] = wp.atomic_add(neighbor_pointer, 0, neighbor_id)


@wp.kernel
def store_neighbor(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_list_index: wp.array(dtype=wp.uint32),
    neighbor_list: wp.array(dtype=wp.uint32),
    neighbor_distance: wp.array(dtype=float),
):
    tid = wp.tid()
    neighbor_id = 0

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    start_index = neighbor_list_index[i]

    # get local particle variables
    x = particle_x[i]

    # store neighbors
    for index in wp.hash_grid_query(grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - particle_x[index])
        if distance < SMOOTHING_LENGTH:
            neighbor_list[start_index + neighbor_id] = index
            neighbor_distance[start_index + neighbor_id] = distance
            neighbor_id += 1


@wp.kernel
def compute_density(
    grid: wp.uint64,
    neighbor_num: wp.array(dtype=wp.uint32),
    neighbor_list_index: wp.array(dtype=wp.uint32),
    neighbor_distance: wp.array(dtype=float),
    particle_rho: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # store density
    rho = float(0.0)

    # loop through neighbors to compute density
    for j in range(neighbor_list_index[i], neighbor_list_index[i] + neighbor_num[i]):
        rho += FLUID_MASS * spline_W(neighbor_distance[j])

    particle_rho[i] = rho


@wp.kernel
def predict_v_adv(
    dt: float,
    particle_v: wp.array(dtype=wp.vec3),
    particle_v_adv: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_v_adv[tid] = particle_v[tid] + wp.vec3(0.0, -GRAVITY, 0.0) * dt


@wp.kernel
def compute_d_ii(
    grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_num: wp.array(dtype=wp.vec3),
    neighbor_list: wp.array(dtype=wp.uint32),
    neighbor_list_index: wp.array(dtype=wp.uint32),
    particle_rho: wp.array(dtype=float),
    term_d_ii: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # store density
    d_ii = 0.0

    # loop through neighbors to compute density
    for j in range(neighbor_list_index[i], neighbor_list_index[i] + neighbor_num[i]):
        index = neighbor_list[j]
        grad_W_ij = grad_spline_W(particle_x[index] - particle_x[i])
        d_ii -= grad_W_ij / particle_rho[i] ** 2

    term_d_ii[i] = d_ii * FLUID_MASS * dt**2


@wp.kernel
def predict_rho_adv(
    grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v_adv: wp.array(dtype=wp.vec3),
    neighbor_num: wp.array(dtype=wp.uint32),
    neighbor_list: wp.array(dtype=wp.uint32),
    neighbor_list_index: wp.array(dtype=wp.uint32),
    particle_rho: wp.array(dtype=float),
    particle_rho_adv: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # store density
    rho_adv = 0.0

    # loop through neighbors to compute density
    for j in range(neighbor_list_index[i], neighbor_list_index[i] + neighbor_num[i]):
        index = neighbor_list[j]
        grad_W_ij = grad_spline_W(particle_x[index] - particle_x[i])
        v_ij_adv = particle_v_adv[i] - particle_v_adv[index]
        rho_adv += wp.dot(v_ij_adv, grad_W_ij)

    particle_rho_adv[i] = particle_rho[i] + rho_adv * FLUID_MASS * dt


@wp.kernel
def init_pressure(
    particle_p: wp.array(dtype=float),
):
    tid = wp.tid()

    particle_p[tid] /= 2


@wp.kernel
def compute_a_ii(
    grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v_adv: wp.array(dtype=wp.vec3),
    neighbor_num: wp.array(dtype=wp.uint32),
    neighbor_list: wp.array(dtype=wp.uint32),
    neighbor_list_index: wp.array(dtype=wp.uint32),
    particle_rho: wp.array(dtype=float),
    particle_rho_adv: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    term_a_ii: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # store density
    a_ii = 0.0

    # loop through neighbors to compute density
    for j in range(neighbor_list_index[i], neighbor_list_index[i] + neighbor_num[i]):
        index = neighbor_list[j]
        grad_W_ij = grad_spline_W(particle_x[index] - particle_x[i])
        a_ii += (
            particle_p[i] / particle_rho[i] ** 2
            + particle_p[index] / particle_rho[index] ** 2
        ) * grad_W_ij

    term_a_ii[i] = a_ii * FLUID_MASS * dt


@wp.kernel
def kick(
    particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float
):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(
    particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float
):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3),
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # grid size
    nr_x = wp.int32(width / 4.0 / SMOOTHING_LENGTH)
    nr_y = wp.int32(height / SMOOTHING_LENGTH)
    nr_z = wp.int32(length / 4.0 / SMOOTHING_LENGTH)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = SMOOTHING_LENGTH * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * SMOOTHING_LENGTH * wp.vec3(
        wp.randn(state), wp.randn(state), wp.randn(state)
    )

    # set position
    particle_x[tid] = pos


class II_solver:
    def __init__(self, stage_path="example_sph.usd", verbose=False):
        self.verbose = verbose

        # render params
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        # simulation params
        self.width = 80.0  # x
        self.height = 80.0  # y
        self.length = 80.0  # z
        self.isotropic_exp = 20
        self.base_density = 1.0
        self.particle_mass = FLUID_MASS
        self.dt = 0.01 * SMOOTHING_LENGTH  # decrease sim dt by smoothing length
        self.dynamic_visc = 0.025
        self.damping_coef = -0.95
        self.n = int(
            self.height
            * (self.width / 4.0)
            * (self.height / 4.0)
            / (SMOOTHING_LENGTH**3)
        )  # number particles (small box in corner)
        self.sim_step_to_frame_ratio = int(32 / SMOOTHING_LENGTH)

        # allocate arrays
        self.x = wp.empty(self.n, dtype=wp.vec3)
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.v_adv = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.rho_adv = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.p = wp.zeros(self.n, dtype=float)
        self.term_d_ii = wp.zeros(self.n, dtype=wp.vec3)
        self.term_a_ii = wp.zeros(self.n, dtype=float)
        self.neighbor_num = wp.zeros(self.n, dtype=wp.uint32)
        self.neighbor_list = wp.zeros(self.n * 100, dtype=wp.uint32)
        self.neighbor_distance = wp.zeros(self.n * 100, dtype=float)
        self.neighbor_list_index = wp.zeros(self.n, dtype=wp.uint32)

        # set random positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[
                self.x,
                self.width,
                self.height,
                self.length,
            ],
        )  # initialize in small area

        # create hash array
        grid_size = int(self.height / (4.0 * SMOOTHING_LENGTH))
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

        # renderer
        self.renderer = wp.render.UsdRenderer(stage_path) if stage_path else None

    def step(self):
        with wp.ScopedTimer("step"):
            for _ in range(self.sim_step_to_frame_ratio):
                with wp.ScopedTimer("neighbor search", active=self.verbose):
                    self.neighbor_search()

                with wp.ScopedTimer("predict advection", active=self.verbose):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.neighbor_num,
                            self.neighbor_list_index,
                            self.neighbor_list,
                            self.x,
                        ],
                        outputs=[self.rho],
                    )

                    # predict advection
                    wp.launch(
                        kernel=predict_v_adv,
                        dim=self.n,
                        inputs=[
                            self.dt,
                            self.v,
                        ],
                        outputs=[self.v_adv],
                    )

                    # compute d_ii
                    wp.launch(
                        kernel=compute_d_ii,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.dt,
                            self.x,
                            self.neighbor_num,
                            self.neighbor_list,
                            self.neighbor_list_index,
                            self.rho,
                        ],
                        outputs=[self.term_d_ii],
                    )

                    # predict density advection
                    wp.launch(
                        kernel=predict_rho_adv,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.dt,
                            self.x,
                            self.v_adv,
                            self.neighbor_num,
                            self.neighbor_list,
                            self.neighbor_list_index,
                            self.rho,
                        ],
                        outputs=[self.rho_adv],
                    )

                    # init pressure
                    wp.launch(
                        kernel=init_pressure,
                        dim=self.n,
                        outputs=[self.p],
                    )

                    # TODO conpute a_ii

                with wp.ScopedTimer("pressure solve", active=self.verbose):
                    pass

                with wp.ScopedTimer("integration", active=self.verbose):
                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(
                        kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt]
                    )

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(),
                radius=SMOOTHING_LENGTH,
                name="points",
                colors=(0.8, 0.3, 0.2),
            )
            self.renderer.end_frame()

    def neighbor_search(self):
        self.grid.build(self.x, SMOOTHING_LENGTH)
        neighbor_pointer = wp.zeros(1, dtype=wp.uint32)

        wp.launch(
            kernel=count_neighbor,
            dim=self.n,
            inputs=[
                self.grid.id,
                self.x,
            ],
            outputs=[
                neighbor_pointer,
                self.neighbor_num,
                self.neighbor_list_index,
            ],
        )

        wp.launch(
            kernel=store_neighbor,
            dim=self.n,
            inputs=[
                self.grid.id,
                self.x,
                self.neighbor_list_index,
            ],
            outputs=[
                self.neighbor_list,
                self.neighbor_distance,
            ],
        )
