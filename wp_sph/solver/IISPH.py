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

import numpy as np
import warp as wp
import warp.render

from ..main import *
from .sph_funcs import *  # TODO cache the functions


@wp.kernel
def initialize_fluid(
    particle_x: wp.array(dtype=wp.vec3),
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # grid size
    nr_x = wp.int32(width / 4.0 / DIAMETER)
    nr_y = wp.int32(height / DIAMETER)
    nr_z = wp.int32(length / 4.0 / DIAMETER)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = DIAMETER * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * DIAMETER * wp.vec3(
        wp.randn(state), wp.randn(state), wp.randn(state)
    )

    # set position
    # TODO remove the offset
    particle_x[tid] = pos + wp.vec3(width / 3.0, height / 4.0, length / 3.0)


def initialize_box(width, height, length, spacing, layers):
    """
    Generate boundary particle positions for a box with exactly the specified number of layers.
    Creates boundary particles around a fluid domain, with open top.
    """
    x_range = range(-layers, int(width / spacing) + layers)
    y_range = range(-layers, int(height / spacing))  # No top boundary
    z_range = range(-layers, int(length / spacing) + layers)

    # Calculate domain bounds (interior region)
    x_min, x_max = 0, int(width / spacing) - 1
    y_min, y_max = 0, int(height / spacing) - 1
    z_min, z_max = 0, int(length / spacing) - 1

    particles = []

    # Generate all candidate positions
    for x_idx in x_range:
        for y_idx in y_range:
            for z_idx in z_range:
                is_outside = (
                    x_idx < x_min
                    or x_idx > x_max
                    or y_idx < y_min  # No condition for y_max to keep top open
                    or z_idx < z_min
                    or z_idx > z_max
                )

                is_within_layers = (
                    x_idx >= x_min - layers
                    and x_idx <= x_max + layers
                    and y_idx >= y_min - layers  # No upper y constraint
                    and z_idx >= z_min - layers
                    and z_idx <= z_max + layers
                )

                if is_outside and is_within_layers:
                    # Convert to physical coordinates
                    x = x_idx * spacing
                    y = y_idx * spacing
                    z = z_idx * spacing
                    particles.append([x, y, z])

    particles_array = wp.array(particles, dtype=wp.vec3)

    return particles_array, len(particles)


@wp.kernel
def compute_boundary_density(
    boundary_grid: wp.uint64,
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_phi: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(boundary_grid, tid)

    # get local particle variables
    x = boundary_x[i]

    # store density
    rho = float(0.0)

    # loop through neighbors to compute density
    for index in wp.hash_grid_query(boundary_grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - boundary_x[index])
        if distance < SMOOTHING_LENGTH:
            rho += spline_W(distance)

    boundary_phi[i] = RHO_0 / rho


@wp.kernel
def count_neighbor(
    grid: wp.uint64,
    neighbor_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_x: wp.array(dtype=wp.vec3),
    neighbor_pointer: wp.array(dtype=int),
    neighbor_num: wp.array(dtype=int),
    neighbor_list_index: wp.array(dtype=int),
):
    tid = wp.tid()
    neighbor_id = wp.int32(0)

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # count neighbors
    for index in wp.hash_grid_query(neighbor_grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - neighbor_x[index])
        if distance < SMOOTHING_LENGTH:
            neighbor_id += 1

    # store number of neighbors
    neighbor_num[i] = neighbor_id
    neighbor_list_index[i] = wp.atomic_add(neighbor_pointer, 0, neighbor_id)


@wp.kernel
def store_neighbor(
    grid: wp.uint64,
    neighbor_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_x: wp.array(dtype=wp.vec3),
    neighbor_list_index: wp.array(dtype=int),
    neighbor_list: wp.array(dtype=int),
    neighbor_distance: wp.array(dtype=float),
):
    tid = wp.tid()
    neighbor_id = wp.int32(0)

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    start_index = neighbor_list_index[i]

    # get local particle variables
    x = particle_x[i]

    # store neighbors
    for index in wp.hash_grid_query(neighbor_grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - neighbor_x[index])
        if distance < SMOOTHING_LENGTH:
            neighbor_list[start_index + neighbor_id] = index
            neighbor_distance[start_index + neighbor_id] = distance
            neighbor_id += 1


@wp.kernel
def init_pressure(
    particle_p: wp.array(dtype=float),
):
    tid = wp.tid()

    particle_p[tid] /= 2.0


@wp.kernel
def compute_density(
    fluid_grid: wp.uint64,
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    ff_neighbor_distance: wp.array(dtype=float),
    fs_neighbor_num: wp.array(dtype=int),
    fs_neighbor_list: wp.array(dtype=int),
    fs_neighbor_list_index: wp.array(dtype=int),
    fs_neighbor_distance: wp.array(dtype=float),
    boundary_phi: wp.array(dtype=float),
    fluid_rho: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)
    term_2 = float(0.0)

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        term_1 += spline_W(ff_neighbor_distance[j])

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        term_2 += spline_W(fs_neighbor_distance[j]) * boundary_phi[fs_neighbor_list[j]]

    fluid_rho[i] = term_1 * FLUID_MASS + term_2


@wp.kernel
def predict_v_adv(
    dt: float,
    particle_v: wp.array(dtype=wp.vec3),
    particle_v_adv: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_v_adv[tid] = particle_v[tid] + wp.vec3(0.0, -GRAVITY, 0.0) * dt


@wp.kernel
def predict_rho_adv(
    fluid_grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),
    boundary_x: wp.array(dtype=wp.vec3),
    particle_v_adv: wp.array(dtype=wp.vec3),
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    fs_neighbor_num: wp.array(dtype=int),
    fs_neighbor_list: wp.array(dtype=int),
    fs_neighbor_list_index: wp.array(dtype=int),
    boundary_phi: wp.array(dtype=float),
    fluid_rho: wp.array(dtype=float),
    fluid_rho_adv: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)
    term_2 = float(0.0)

    x = particle_x[i]
    v_adv = particle_v_adv[i]

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        v_ij = v_adv - particle_v_adv[index]
        nabla_W_ij = grad_spline_W(particle_x[index] - x)
        term_1 += wp.dot(v_ij, nabla_W_ij)

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        nabla_W_ib = grad_spline_W(boundary_x[fs_neighbor_list[j]] - x)
        term_2 += wp.dot(v_adv, nabla_W_ib) * boundary_phi[fs_neighbor_list[j]]

    fluid_rho_adv[i] = fluid_rho[i] + (term_1 * FLUID_MASS + term_2) * dt


@wp.kernel
def compute_term_d(  # TODO check again
    fluid_grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),
    boundary_x: wp.array(dtype=wp.vec3),
    fluid_rho: wp.array(dtype=float),
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    fs_neighbor_num: wp.array(dtype=int),
    fs_neighbor_list: wp.array(dtype=int),
    fs_neighbor_list_index: wp.array(dtype=int),
    boundary_phi: wp.array(dtype=float),
    term_d_ii: wp.array(dtype=wp.vec3),
    term_d_ij: wp.array(dtype=wp.vec3),
    term_d_ji: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = wp.vec3(0.0, 0.0, 0.0)
    term_2 = wp.vec3(0.0, 0.0, 0.0)
    term_3 = FLUID_MASS * dt**2.0

    x = particle_x[i]
    rho = fluid_rho[i]

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        grad_W_ij = grad_spline_W(particle_x[index] - x)
        term_1 -= grad_W_ij / rho**2.0

        term_d_ij[j] = -grad_W_ij * term_3 / fluid_rho[index] ** 2.0
        term_d_ji[j] = grad_W_ij * term_3 / rho**2.0

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        grad_W_ib = grad_spline_W(boundary_x[fs_neighbor_list[j]] - x)
        term_2 -= grad_W_ib * boundary_phi[fs_neighbor_list[j]] / rho**2.0

    term_d_ii[i] = (term_1 * FLUID_MASS + term_2) * dt**2.0


@wp.kernel
def compute_term_a(
    fluid_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    boundary_x: wp.array(dtype=wp.vec3),
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    fs_neighbor_num: wp.array(dtype=int),
    fs_neighbor_list: wp.array(dtype=int),
    fs_neighbor_list_index: wp.array(dtype=int),
    boundary_phi: wp.array(dtype=float),
    term_d_ii: wp.array(dtype=wp.vec3),
    term_d_ji: wp.array(dtype=wp.vec3),
    term_a_ii: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)
    term_2 = float(0.0)

    x = particle_x[i]
    d_ii = term_d_ii[i]

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        grad_W_ij = grad_spline_W(particle_x[ff_neighbor_list[j]] - x)
        term_1 += wp.dot((d_ii - term_d_ji[j]), grad_W_ij)

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        grad_W_ib = grad_spline_W(boundary_x[fs_neighbor_list[j]] - x)
        term_2 += wp.dot(d_ii, grad_W_ib) * boundary_phi[fs_neighbor_list[j]]

    term_a_ii[i] = term_1 * FLUID_MASS + term_2


@wp.kernel
def compute_term_Ap(  # TODO check again
    fluid_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_p: wp.array(dtype=float),
    boundary_x: wp.array(dtype=wp.vec3),
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    fs_neighbor_num: wp.array(dtype=int),
    fs_neighbor_list: wp.array(dtype=int),
    fs_neighbor_list_index: wp.array(dtype=int),
    boundary_phi: wp.array(dtype=float),
    term_d_ii: wp.array(dtype=wp.vec3),
    term_d_ij: wp.array(dtype=wp.vec3),
    term_d_ji: wp.array(dtype=wp.vec3),
    term_a_ii: wp.array(dtype=float),
    sum_d_ij_p_j: wp.array(dtype=wp.vec3),
    term_Ap_i: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)
    term_2 = float(0.0)

    x = particle_x[i]
    sum_d_ij_p_j[i] = wp.vec3(0.0)

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        d_jk_p_k = wp.vec3(0.0)
        for k in range(
            ff_neighbor_list_index[index],
            ff_neighbor_list_index[index] + ff_neighbor_num[index],
        ):
            d_jk_p_k += term_d_ij[k] * particle_p[ff_neighbor_list[k]]

        term = (
            term_d_ji[j] * particle_p[i]
            - term_d_ii[index] * particle_p[index]
            - d_jk_p_k
        )

        term_1 += wp.dot(term, grad_spline_W(particle_x[index] - x))

        sum_d_ij_p_j[i] += term_d_ij[j] * particle_p[index]

    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        term_1 += wp.dot(
            sum_d_ij_p_j[i], grad_spline_W(particle_x[ff_neighbor_list[j]] - x)
        )

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        term_2 += (
            wp.dot(sum_d_ij_p_j[i], grad_spline_W(boundary_x[fs_neighbor_list[j]] - x))
            * boundary_phi[fs_neighbor_list[j]]
        )

    term_Ap_i[i] = term_1 * FLUID_MASS + term_2 + term_a_ii[i] * particle_p[i]


@wp.kernel
def update_p_rho(
    fluid_grid: wp.uint64,
    fluid_rho_adv: wp.array(dtype=float),
    term_a_ii: wp.array(dtype=float),
    term_Ap_i: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    sum_rho: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    updated_rho = fluid_rho_adv[i] + term_Ap_i[i]
    wp.atomic_add(sum_rho, 0, updated_rho)

    # TODO check this clamp
    term_a_ii_i = term_a_ii[i]
    if term_a_ii_i > -INV_SMALL and term_a_ii_i < INV_SMALL:
        term_a_ii_i = wp.sign(term_a_ii_i) * INV_SMALL

    particle_p[i] += (RHO_0 - updated_rho) * OMEGA / term_a_ii_i
    particle_p[i] = wp.max(particle_p[i], 0.0)


@wp.kernel
def kick(
    inv_dt: float,
    particle_p: wp.array(dtype=float),
    particle_v_adv: wp.array(dtype=wp.vec3),
    term_d_ii: wp.array(dtype=wp.vec3),
    sum_d_ij_p_j: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_v_max: wp.array(dtype=float),
):
    tid = wp.tid()
    v = particle_v_adv[tid] + inv_dt * (
        term_d_ii[tid] * particle_p[tid] + sum_d_ij_p_j[tid]
    )
    particle_v[tid] = v
    wp.atomic_max(particle_v_max, 0, wp.length(v))


@wp.kernel
def drift(
    dt: float,
    particle_v: wp.array(dtype=wp.vec3),
    particle_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_x[tid] += particle_v[tid] * dt


class IISPH:
    def __init__(self, stage_path="example_sph.usd", verbose=False):
        self.verbose = verbose

        # render params
        # fps = 100
        # self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        # simulation params
        self.width = 2.0  # x
        self.height = 2.0  # y
        self.length = 2.0  # z
        self.dt = TIME_STEP_MAX
        self.inv_dt = 1 / self.dt
        self.boundary_layer = 3
        self.n = int(
            self.height * (self.width / 4.0) * (self.height / 4.0) / (DIAMETER**3)
        )  # number particles (small box in corner)

        # set boundary
        self.boundary_x, self.boundary_n = initialize_box(
            self.width,
            self.height,
            self.length,
            DIAMETER,
            self.boundary_layer,
        )
        self.boundary_phi = wp.zeros(self.boundary_n, dtype=float)
        self.boundary_grid = wp.HashGrid(
            int(self.width / SMOOTHING_LENGTH),
            int(self.height / SMOOTHING_LENGTH),
            int(self.length / SMOOTHING_LENGTH),
        )
        self.boundary_grid.build(self.boundary_x, SMOOTHING_LENGTH)

        # compute PHI value of boundary particles
        wp.launch(
            kernel=compute_boundary_density,
            dim=self.boundary_n,
            inputs=[
                self.boundary_grid.id,
                self.boundary_x,
            ],
            outputs=[self.boundary_phi],
        )

        # allocate arrays
        self.x = wp.empty(self.n, dtype=wp.vec3)
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.v_adv = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.rho_adv = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.p = wp.zeros(self.n, dtype=float)
        self.sum_rho = wp.zeros(1, dtype=float)
        self.term_a_ii = wp.zeros(self.n, dtype=float)
        self.term_d_ii = wp.zeros(self.n, dtype=wp.vec3)
        self.term_d_ij = wp.zeros(self.n * 100, dtype=wp.vec3)
        self.term_d_ji = wp.zeros(self.n * 100, dtype=wp.vec3)
        self.term_Ap_i = wp.zeros(self.n, dtype=float)
        self.sum_d_ij_p_j = wp.zeros(self.n, dtype=wp.vec3)
        self.ff_neighbor_num = wp.zeros(self.n, dtype=int)
        self.ff_neighbor_list = wp.zeros(self.n * 100, dtype=int)
        self.ff_neighbor_distance = wp.zeros(self.n * 100, dtype=float)
        self.ff_neighbor_list_index = wp.zeros(self.n, dtype=int)
        self.fs_neighbor_num = wp.zeros(self.n, dtype=int)
        self.fs_neighbor_list = wp.zeros(self.n * 100, dtype=int)
        self.fs_neighbor_distance = wp.zeros(self.n * 100, dtype=float)
        self.fs_neighbor_list_index = wp.zeros(self.n, dtype=int)

        # set fluid
        wp.launch(
            kernel=initialize_fluid,
            dim=self.n,
            inputs=[
                self.x,
                self.width,
                self.height,
                self.length,
            ],
        )

        # create hash array
        self.fluid_grid = wp.HashGrid(
            int(self.width / SMOOTHING_LENGTH),
            int(self.height / SMOOTHING_LENGTH),
            int(self.length / SMOOTHING_LENGTH),
        )

        # renderer
        self.renderer = wp.render.UsdRenderer(stage_path) if stage_path else None

    def step(self):  # TODO use CUDA graph capture
        with wp.ScopedTimer("step"):
            with wp.ScopedTimer("neighbor search", active=self.verbose):
                self.neighbor_search()

            with wp.ScopedTimer("predict advection", active=self.verbose):

                # init pressure
                wp.launch(
                    kernel=init_pressure,
                    dim=self.n,
                    outputs=[self.p],
                )

                # compute density
                wp.launch(
                    kernel=compute_density,
                    dim=self.n,
                    inputs=[
                        self.fluid_grid.id,
                        self.ff_neighbor_num,
                        self.ff_neighbor_list_index,
                        self.ff_neighbor_distance,
                        self.fs_neighbor_num,
                        self.fs_neighbor_list,
                        self.fs_neighbor_list_index,
                        self.fs_neighbor_distance,
                        self.boundary_phi,
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

                # predict density advection
                wp.launch(
                    kernel=predict_rho_adv,
                    dim=self.n,
                    inputs=[
                        self.fluid_grid.id,
                        self.dt,
                        self.x,
                        self.boundary_x,
                        self.v_adv,
                        self.ff_neighbor_num,
                        self.ff_neighbor_list,
                        self.ff_neighbor_list_index,
                        self.fs_neighbor_num,
                        self.fs_neighbor_list,
                        self.fs_neighbor_list_index,
                        self.boundary_phi,
                        self.rho,
                    ],
                    outputs=[self.rho_adv],
                )

                # compute d_ii, d_ij, d_ji
                wp.launch(
                    kernel=compute_term_d,
                    dim=self.n,
                    inputs=[
                        self.fluid_grid.id,
                        self.dt,
                        self.x,
                        self.boundary_x,
                        self.rho,
                        self.ff_neighbor_num,
                        self.ff_neighbor_list,
                        self.ff_neighbor_list_index,
                        self.fs_neighbor_num,
                        self.fs_neighbor_list,
                        self.fs_neighbor_list_index,
                        self.boundary_phi,
                    ],
                    outputs=[
                        self.term_d_ii,
                        self.term_d_ij,
                        self.term_d_ji,
                    ],
                )

                # compute a_ii
                wp.launch(
                    kernel=compute_term_a,
                    dim=self.n,
                    inputs=[
                        self.fluid_grid.id,
                        self.x,
                        self.boundary_x,
                        self.ff_neighbor_num,
                        self.ff_neighbor_list,
                        self.ff_neighbor_list_index,
                        self.fs_neighbor_num,
                        self.fs_neighbor_list,
                        self.fs_neighbor_list_index,
                        self.boundary_phi,
                        self.term_d_ii,
                        self.term_d_ji,
                    ],
                    outputs=[self.term_a_ii],
                )

            with wp.ScopedTimer("pressure solve", active=self.verbose):
                loop = 0
                while (loop < 2) or (self.average_rho - RHO_0 > ETA):
                    if loop > 100:
                        self.raise_error("Pressure solver did not converge.")

                    # solve pressure
                    wp.launch(
                        kernel=compute_term_Ap,
                        dim=self.n,
                        inputs=[
                            self.fluid_grid.id,
                            self.x,
                            self.p,
                            self.boundary_x,
                            self.ff_neighbor_num,
                            self.ff_neighbor_list,
                            self.ff_neighbor_list_index,
                            self.fs_neighbor_num,
                            self.fs_neighbor_list,
                            self.fs_neighbor_list_index,
                            self.boundary_phi,
                            self.term_d_ii,
                            self.term_d_ij,
                            self.term_d_ji,
                            self.term_a_ii,
                        ],
                        outputs=[
                            self.sum_d_ij_p_j,
                            self.term_Ap_i,
                        ],
                    )

                    self.sum_rho = wp.zeros(1, dtype=float)
                    wp.launch(
                        kernel=update_p_rho,
                        dim=self.n,
                        inputs=[
                            self.fluid_grid.id,
                            self.rho_adv,
                            self.term_a_ii,
                            self.term_Ap_i,
                        ],
                        outputs=[
                            self.p,
                            self.sum_rho,
                        ],
                    )

                    loop += 1

            with wp.ScopedTimer("integration", active=self.verbose):
                v_max = wp.zeros(1, dtype=float)
                # kick
                wp.launch(
                    kernel=kick,
                    dim=self.n,
                    inputs=[
                        self.inv_dt,
                        self.p,
                        self.v_adv,
                        self.term_d_ii,
                        self.sum_d_ij_p_j,
                    ],
                    outputs=[self.v, v_max],
                )

                # drift
                wp.launch(
                    kernel=drift,
                    dim=self.n,
                    inputs=[self.dt, self.v],
                    outputs=[self.x],
                )

            self.sim_time += self.dt
            self.dt = wp.min(  # CFL condition
                0.4 * SMOOTHING_LENGTH / v_max.numpy()[0], TIME_STEP_MAX
            )
            self.inv_dt = 1 / self.dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(),
                radius=DIAMETER / 2.0,
                name="fluid",
                colors=(0.5, 0.5, 0.8),
            )
            self.renderer.render_points(
                points=self.boundary_x.numpy(),
                radius=wp.constant(DIAMETER / 1.4),
                name="boundary",
                colors=(0.6, 0.7, 0.8),
            )
            self.renderer.end_frame()

    def neighbor_search(self):
        """
        Neighbor search for both fluid-fluid and fluid-boundary interactions.
        """

        # build grid of fluid particles
        self.fluid_grid.build(self.x, SMOOTHING_LENGTH)

        # search fluid neighbors for fluid
        neighbor_pointer = wp.zeros(1, dtype=int)

        wp.launch(
            kernel=count_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.fluid_grid.id,
                self.x,
                self.x,
            ],
            outputs=[
                neighbor_pointer,
                self.ff_neighbor_num,
                self.ff_neighbor_list_index,
            ],
        )

        print(
            f"Completed search for fluid neighbors of fluid. Cached {neighbor_pointer.numpy()[0]} neighbors in array of size {self.ff_neighbor_list.shape[0]}."
        )

        wp.launch(
            kernel=store_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.fluid_grid.id,
                self.x,
                self.x,
                self.ff_neighbor_list_index,
            ],
            outputs=[
                self.ff_neighbor_list,
                self.ff_neighbor_distance,
            ],
        )

        # search boundary neighbors for fluid
        neighbor_pointer = wp.zeros(1, dtype=int)

        wp.launch(
            kernel=count_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.boundary_grid.id,
                self.x,
                self.boundary_x,
            ],
            outputs=[
                neighbor_pointer,
                self.fs_neighbor_num,
                self.fs_neighbor_list_index,
            ],
        )

        print(
            f"Completed search for boundary neighbors of fluid. Cached {neighbor_pointer.numpy()[0]} neighbors in array of size {self.fs_neighbor_list.shape[0]}."
        )

        wp.launch(
            kernel=store_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.boundary_grid.id,
                self.x,
                self.boundary_x,
                self.fs_neighbor_list_index,
            ],
            outputs=[
                self.fs_neighbor_list,
                self.fs_neighbor_distance,
            ],
        )

    def raise_error(self, message):
        if self.renderer:
            self.renderer.save()
        raise RuntimeError(message)

    @property
    def average_rho(self):
        return self.sum_rho.numpy()[0] / self.n
