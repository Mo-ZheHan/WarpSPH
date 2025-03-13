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


def initialize_fluid(min_point, max_point, spacing):
    """
    Generate fluid particles in a rectangular region using numpy's meshgrid.
    """
    assert 0 < min_point[0] < max_point[0] < BOX_WIDTH
    assert 0 < min_point[1] < max_point[1] < BOX_HEIGHT
    assert 0 < min_point[2] < max_point[2] < BOX_LENGTH

    x = np.arange(min_point[0], max_point[0], spacing)
    y = np.arange(min_point[1], max_point[1], spacing)
    z = np.arange(min_point[2], max_point[2], spacing)

    # Create 3D grid of positions
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Convert to warp array
    particles_array = wp.array(positions, dtype=wp.vec3)

    return particles_array, positions.shape[0]


def initialize_box(layers, spacing):
    """
    Generate boundary particle positions for a box with exactly the specified number of layers.
    """
    x_range = range(-layers, int(BOX_WIDTH / spacing) + layers)
    y_range = range(-layers, int(BOX_HEIGHT / spacing) + layers)
    z_range = range(-layers, int(BOX_LENGTH / spacing) + layers)

    # Calculate domain bounds (interior region)
    x_min, x_max = 0, int(BOX_WIDTH / spacing) - 1
    y_min, y_max = 0, int(BOX_HEIGHT / spacing) - 1
    z_min, z_max = 0, int(BOX_LENGTH / spacing) - 1

    particles = []

    # Generate all candidate positions
    for x_idx in x_range:
        for y_idx in y_range:
            for z_idx in z_range:
                is_outside = (
                    x_idx < x_min
                    or x_idx > x_max
                    or y_idx < y_min
                    or y_idx > y_max
                    or z_idx < z_min
                    or z_idx > z_max
                )

                is_within_layers = (
                    x_idx >= x_min - layers
                    and x_idx <= x_max + layers
                    and y_idx >= y_min - layers
                    and y_idx <= y_max + layers
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

    # # order threads by cell
    # i = wp.hash_grid_point_id(boundary_grid, tid)

    # # get local particle variables
    # x = boundary_x[i]

    # # store density
    # rho = spline_W(0.0)

    # # loop through neighbors to compute density
    # for index in wp.hash_grid_query(boundary_grid, x, SMOOTHING_LENGTH):
    #     distance = wp.length(x - boundary_x[index])
    #     if distance < SMOOTHING_LENGTH and index != i:
    #         rho += spline_W(distance)

    # boundary_phi[i] = RHO_0 / rho

    # TODO check this
    boundary_phi[tid] = FLUID_MASS


@wp.kernel
def count_same_neighbor(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
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
    for index in wp.hash_grid_query(grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - particle_x[index])
        if distance < SMOOTHING_LENGTH and index != i:
            neighbor_id += 1

    # store number of neighbors
    neighbor_num[i] = neighbor_id
    neighbor_list_index[i] = wp.atomic_add(neighbor_pointer, 0, neighbor_id)


@wp.kernel
def store_same_neighbor(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_num: wp.array(dtype=int),
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
    for index in wp.hash_grid_query(grid, x, SMOOTHING_LENGTH):
        distance = wp.length(x - particle_x[index])
        if distance < SMOOTHING_LENGTH and index != i:
            neighbor_list[start_index + neighbor_id] = index
            neighbor_distance[start_index + neighbor_id] = distance
            neighbor_id += 1

    # TODO remove this check
    if neighbor_id != neighbor_num[i]:
        print("ERROR: Neighbor count mismatch.")


@wp.kernel
def count_diff_neighbor(
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
def store_diff_neighbor(
    grid: wp.uint64,
    neighbor_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    neighbor_x: wp.array(dtype=wp.vec3),
    neighbor_num: wp.array(dtype=int),
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

    # TODO remove this check
    if neighbor_id != neighbor_num[i]:
        print("ERROR: Neighbor count mismatch.")


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

    # TODO check this clamp
    fluid_rho[i] = term_1 * FLUID_MASS + term_2 + spline_W(0.0) * FLUID_MASS
    # fluid_rho[i] = wp.max(
    #     term_1 * FLUID_MASS + term_2 + spline_W(0.0) * FLUID_MASS, RHO_0
    # )


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

    rho_adv = fluid_rho[i] + (term_1 * FLUID_MASS + term_2) * dt
    fluid_rho_adv[i] = wp.max(rho_adv, RHO_0)  # TODO check this clamp


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
        term_1 -= grad_W_ij

        term_d_ij[j] = -grad_W_ij * term_3 / fluid_rho[index] ** 2.0
        term_d_ji[j] = grad_W_ij * term_3 / rho**2.0

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        grad_W_ib = grad_spline_W(boundary_x[index] - x)
        term_2 -= grad_W_ib * boundary_phi[index]

    term_d_ii[i] = (term_1 * FLUID_MASS + term_2) * dt**2.0 / rho**2.0


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
        index = fs_neighbor_list[j]
        grad_W_ib = grad_spline_W(boundary_x[index] - x)
        term_2 += wp.dot(d_ii, grad_W_ib) * boundary_phi[index]

    term_a_ii[i] = term_1 * FLUID_MASS + term_2


@wp.kernel
def compute_term_Ap_1(  # TODO check again
    fluid_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_p: wp.array(dtype=float),
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    term_d_ii: wp.array(dtype=wp.vec3),
    term_d_ij: wp.array(dtype=wp.vec3),
    term_d_ji: wp.array(dtype=wp.vec3),
    sum_d_ij_p_j: wp.array(dtype=wp.vec3),
    term_Ap_i: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)

    x = particle_x[i]
    sum_d_ij_p_j[i] = wp.vec3(0.0)

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        term = term_d_ji[j] * particle_p[i] - term_d_ii[index] * particle_p[index]
        term_1 += wp.dot(term, grad_spline_W(particle_x[index] - x))
        sum_d_ij_p_j[i] += term_d_ij[j] * particle_p[index]

    term_Ap_i[i] = term_1 * FLUID_MASS


@wp.kernel
def compute_term_Ap_2(  # TODO check again
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
    sum_d_ij_p_j_i = sum_d_ij_p_j[i]

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        term_1 += wp.dot(
            sum_d_ij_p_j_i - sum_d_ij_p_j[index], grad_spline_W(particle_x[index] - x)
        )

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        term_2 += (
            wp.dot(sum_d_ij_p_j_i, grad_spline_W(boundary_x[index] - x))
            * boundary_phi[index]
        )

    term_Ap_i[i] += term_1 * FLUID_MASS + term_2 + term_a_ii[i] * particle_p[i]


@wp.kernel
def update_p_rho(
    fluid_grid: wp.uint64,
    fluid_rho_adv: wp.array(dtype=float),
    term_a_ii: wp.array(dtype=float),
    term_Ap_i: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    sum_rho_error: wp.array(dtype=float),
    new_rho: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    updated_rho = fluid_rho_adv[i] + term_Ap_i[i]
    new_rho[i] = updated_rho  # TODO remove this
    eta = wp.abs(updated_rho / RHO_0 - 1.0)
    wp.atomic_add(sum_rho_error, 0, eta)

    term_a_ii_i = term_a_ii[i]
    wp.clamp(term_a_ii_i, -INV_SMALL, INV_SMALL)

    # TODO check this clamp
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
    delta_v_p: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    delta_v = inv_dt * (term_d_ii[tid] * particle_p[tid] + sum_d_ij_p_j[tid])
    v = particle_v_adv[tid] + delta_v
    particle_v[tid] = v
    delta_v_p[tid] = delta_v
    wp.atomic_max(particle_v_max, 0, wp.length(v))


# TODO remove this check
@wp.kernel
def check_rho_update(
    fluid_grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),
    delta_v_p: wp.array(dtype=wp.vec3),
    boundary_x: wp.array(dtype=wp.vec3),
    ff_neighbor_num: wp.array(dtype=int),
    ff_neighbor_list: wp.array(dtype=int),
    ff_neighbor_list_index: wp.array(dtype=int),
    fs_neighbor_num: wp.array(dtype=int),
    fs_neighbor_list: wp.array(dtype=int),
    fs_neighbor_list_index: wp.array(dtype=int),
    boundary_phi: wp.array(dtype=float),
    fluid_rho_adv: wp.array(dtype=float),
    new_rho: wp.array(dtype=float),
    rho_error_check: wp.array(dtype=float),
    rho_error_check_max: wp.array(dtype=float),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)
    term_2 = float(0.0)

    x_i = particle_x[i]
    delta_v_i = delta_v_p[i]

    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        grad_W_ij = grad_spline_W(particle_x[index] - x_i)
        term_1 += wp.dot(delta_v_i - delta_v_p[index], grad_W_ij)

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        grad_W_ib = grad_spline_W(boundary_x[index] - x_i)
        term_2 += wp.dot(delta_v_i, grad_W_ib) * boundary_phi[index]

    updated_rho = fluid_rho_adv[i] + (term_1 * FLUID_MASS + term_2) * dt
    delta = wp.abs(updated_rho - new_rho[i])
    rho_error_check[i] = delta
    wp.atomic_max(rho_error_check_max, 0, delta)


@wp.kernel
def drift(
    dt: float,
    particle_v: wp.array(dtype=wp.vec3),
    particle_x: wp.array(dtype=wp.vec3),
    penetration_times: wp.array(dtype=int),
):
    tid = wp.tid()
    new_pos = particle_x[tid] + particle_v[tid] * dt

    # TODO remove this check
    # penetration_detected = False

    # if new_pos[0] < -3.0 * DIAMETER:
    #     new_pos[0] = DIAMETER
    #     particle_v[tid] = wp.vec3(0.0, particle_v[tid][1], particle_v[tid][2])
    #     penetration_detected = True
    # elif new_pos[0] > BOX_WIDTH + 3.0 * DIAMETER:
    #     new_pos[0] = BOX_WIDTH - DIAMETER
    #     particle_v[tid] = wp.vec3(0.0, particle_v[tid][1], particle_v[tid][2])
    #     penetration_detected = True

    # if new_pos[1] < -3.0 * DIAMETER:
    #     new_pos[1] = DIAMETER
    #     particle_v[tid] = wp.vec3(particle_v[tid][0], 0.0, particle_v[tid][2])
    #     penetration_detected = True
    # elif new_pos[1] > BOX_HEIGHT + 3.0 * DIAMETER:
    #     new_pos[1] = BOX_HEIGHT - DIAMETER
    #     particle_v[tid] = wp.vec3(particle_v[tid][0], 0.0, particle_v[tid][2])
    #     penetration_detected = True

    # if new_pos[2] < -3.0 * DIAMETER:
    #     new_pos[2] = DIAMETER
    #     particle_v[tid] = wp.vec3(particle_v[tid][0], particle_v[tid][1], 0.0)
    #     penetration_detected = True
    # elif new_pos[2] > BOX_LENGTH + 3.0 * DIAMETER:
    #     new_pos[2] = BOX_LENGTH - DIAMETER
    #     particle_v[tid] = wp.vec3(particle_v[tid][0], particle_v[tid][1], 0.0)
    #     penetration_detected = True

    particle_x[tid] = new_pos
    # if penetration_detected:
    #     wp.atomic_add(penetration_times, 0, 1)


class IISPH:
    def __init__(self, stage_path="example_sph.usd", verbose=False):
        self.verbose = verbose

        # render params
        # fps = 100
        # self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        # simulation params
        self.dt = TIME_STEP_MAX
        self.inv_dt = 1 / self.dt
        self.boundary_layer = 3

        # set fluid
        min_point = (BOX_WIDTH * 0.2, BOX_HEIGHT * 0.02, BOX_LENGTH * 0.2)
        max_point = (BOX_WIDTH * 0.8, BOX_HEIGHT * 0.1, BOX_LENGTH * 0.8)
        self.x, self.n = initialize_fluid(min_point, max_point, DIAMETER)

        # set boundary
        self.boundary_x, self.boundary_n = initialize_box(self.boundary_layer, DIAMETER)

        # create hash array
        self.fluid_grid = wp.HashGrid(
            int(BOX_WIDTH / SMOOTHING_LENGTH),
            int(BOX_HEIGHT / SMOOTHING_LENGTH),
            int(BOX_LENGTH / SMOOTHING_LENGTH),
        )
        self.boundary_grid = wp.HashGrid(
            int(BOX_WIDTH / SMOOTHING_LENGTH),
            int(BOX_HEIGHT / SMOOTHING_LENGTH),
            int(BOX_LENGTH / SMOOTHING_LENGTH),
        )

        # allocate arrays
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.v_adv = wp.zeros(self.n, dtype=wp.vec3)
        self.delta_v_p = wp.zeros(self.n, dtype=wp.vec3)  # TODO remove this
        self.rho = wp.zeros(self.n, dtype=float)
        self.rho_adv = wp.zeros(self.n, dtype=float)
        self.new_rho = wp.zeros(self.n, dtype=float)  # TODO remove this
        self.rho_error_check = wp.zeros(self.n, dtype=float)  # TODO remove this
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.p = wp.zeros(self.n, dtype=float)
        self.sum_rho_error = wp.zeros(1, dtype=float)
        self.term_a_ii = wp.zeros(self.n, dtype=float)
        self.term_d_ii = wp.zeros(self.n, dtype=wp.vec3)
        self.term_d_ij = wp.zeros(self.n * 60, dtype=wp.vec3)
        self.term_d_ji = wp.zeros(self.n * 60, dtype=wp.vec3)
        self.term_Ap_i = wp.zeros(self.n, dtype=float)
        self.sum_d_ij_p_j = wp.zeros(self.n, dtype=wp.vec3)
        self.ff_neighbor_num = wp.zeros(self.n, dtype=int)
        self.ff_neighbor_list = wp.zeros(self.n * 60, dtype=int)
        self.ff_neighbor_distance = wp.zeros(self.n * 60, dtype=float)
        self.ff_neighbor_list_index = wp.zeros(self.n, dtype=int)
        self.fs_neighbor_num = wp.zeros(self.n, dtype=int)
        self.fs_neighbor_list = wp.zeros(self.n * 60, dtype=int)
        self.fs_neighbor_distance = wp.zeros(self.n * 60, dtype=float)
        self.fs_neighbor_list_index = wp.zeros(self.n, dtype=int)
        self.boundary_phi = wp.zeros(self.boundary_n, dtype=float)
        self.penetration_times = wp.zeros(1, dtype=int)  # TODO remove this

        # compute PHI value of boundary particles
        self.boundary_grid.build(self.boundary_x, SMOOTHING_LENGTH)
        wp.launch(
            kernel=compute_boundary_density,
            dim=self.boundary_n,
            inputs=[
                self.boundary_grid.id,
                self.boundary_x,
            ],
            outputs=[self.boundary_phi],
        )

        # renderer
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

        if MODE == Mode.DEBUG:
            previewer = wp.render.OpenGLRenderer(
                scaling=4,
                screen_width=1200,
                screen_height=1200,
                draw_axis=False,
                camera_pos=(2, 2, 9.5),
            )

            def custom_input_processor(key_handler):
                import pyglet

                if key_handler[pyglet.window.key.E]:
                    previewer._camera_pos += previewer._camera_up * (
                        previewer._camera_speed * previewer._frame_speed
                    )
                    previewer.update_view_matrix()
                if key_handler[pyglet.window.key.Q]:
                    previewer._camera_pos -= previewer._camera_up * (
                        previewer._camera_speed * previewer._frame_speed
                    )
                    previewer.update_view_matrix()

            previewer.register_input_processor(custom_input_processor)
            self.previewer = previewer
        else:
            self.previewer = None

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
                while (loop < 2) or (self.average_rho_error > ETA):
                    if loop > 100:
                        self.raise_error("Pressure solver did not converge.")

                    # solve pressure
                    wp.launch(
                        kernel=compute_term_Ap_1,
                        dim=self.n,
                        inputs=[
                            self.fluid_grid.id,
                            self.x,
                            self.p,
                            self.ff_neighbor_num,
                            self.ff_neighbor_list,
                            self.ff_neighbor_list_index,
                            self.term_d_ii,
                            self.term_d_ij,
                            self.term_d_ji,
                        ],
                        outputs=[
                            self.sum_d_ij_p_j,
                            self.term_Ap_i,
                        ],
                    )

                    wp.launch(
                        kernel=compute_term_Ap_2,
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
                            self.term_a_ii,
                            self.sum_d_ij_p_j,
                        ],
                        outputs=[
                            self.term_Ap_i,
                        ],
                    )

                    self.sum_rho_error = wp.zeros(1, dtype=float)
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
                            self.sum_rho_error,
                            self.new_rho,
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
                    outputs=[
                        self.v,
                        v_max,
                        self.delta_v_p,
                    ],
                )

                # TODO remove this check
                if MODE == Mode.DEBUG:
                    rho_error_check_max = wp.zeros(1, dtype=float)
                    wp.launch(
                        kernel=check_rho_update,
                        dim=self.n,
                        inputs=[
                            self.fluid_grid.id,
                            self.dt,
                            self.x,
                            self.delta_v_p,
                            self.boundary_x,
                            self.ff_neighbor_num,
                            self.ff_neighbor_list,
                            self.ff_neighbor_list_index,
                            self.fs_neighbor_num,
                            self.fs_neighbor_list,
                            self.fs_neighbor_list_index,
                            self.boundary_phi,
                            self.rho_adv,
                            self.new_rho,
                        ],
                        outputs=[
                            self.rho_error_check,
                            rho_error_check_max,
                        ],
                    )
                    rho_error_max = rho_error_check_max.numpy()[0]

                    if rho_error_max > wp.constant(1e-2 * ETA):
                        self.previewer.paused = True
                        print(f"Rho error: {rho_error_max}")

                # drift
                wp.launch(
                    kernel=drift,
                    dim=self.n,
                    inputs=[self.dt, self.v],
                    outputs=[
                        self.x,
                        self.penetration_times,
                    ],
                )

            self.sim_time += self.dt
            self.dt = wp.min(  # CFL condition
                0.4 * DIAMETER / v_max.numpy()[0], TIME_STEP_MAX
            )
            self.inv_dt = 1 / self.dt

    def activate_renderer(self, renderer):
        renderer.begin_frame(self.sim_time)
        renderer.render_points(
            points=self.x.numpy(),
            radius=DIAMETER / 2.0,
            name="fluid",
            colors=(0.5, 0.5, 0.8),
        )
        # renderer.render_points(
        #     points=self.boundary_x.numpy(),
        #     radius=wp.constant(DIAMETER / 1.4),
        #     name="boundary",
        #     colors=(0.6, 0.7, 0.8),
        # )
        renderer.end_frame()

    def render(self):
        if self.previewer:
            self.activate_renderer(self.previewer)

        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.activate_renderer(self.renderer)

    def neighbor_search(self):
        """
        Neighbor search for both fluid-fluid and fluid-boundary interactions.
        """

        # build grid of fluid particles
        self.fluid_grid.build(self.x, SMOOTHING_LENGTH)

        # search fluid neighbors for fluid
        neighbor_pointer = wp.zeros(1, dtype=int)

        wp.launch(
            kernel=count_same_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.x,
            ],
            outputs=[
                neighbor_pointer,
                self.ff_neighbor_num,
                self.ff_neighbor_list_index,
            ],
        )

        # TODO remove this print
        # print(
        #     f"Completed search for fluid neighbors of fluid. Cached {neighbor_pointer.numpy()[0]} neighbors in array of size {self.ff_neighbor_list.shape[0]}."
        # )

        wp.launch(
            kernel=store_same_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.x,
                self.ff_neighbor_num,
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
            kernel=count_diff_neighbor,
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

        # TODO remove this print
        # print(
        #     f"Completed search for boundary neighbors of fluid. Cached {neighbor_pointer.numpy()[0]} neighbors in array of size {self.fs_neighbor_list.shape[0]}."
        # )

        wp.launch(
            kernel=store_diff_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.boundary_grid.id,
                self.x,
                self.boundary_x,
                self.fs_neighbor_num,
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
    def average_rho_error(self):
        return self.sum_rho_error.numpy()[0] / self.n

    @property
    def window_closed(self):
        if MODE == Mode.DEBUG:
            return self.previewer.has_exit
        else:
            return False
