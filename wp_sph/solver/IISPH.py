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
from .model import load_model
from .sph_funcs import W_table, grad_spline_W, grad_W_table, spline_W


@wp.kernel
def compute_boundary_density(
    W_table: wp.array(dtype=wp.float32),  # type: ignore
    boundary_grid: wp.uint64,
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

    # order threads by cell
    i = wp.hash_grid_point_id(boundary_grid, tid)

    # get local particle variables
    x = boundary_x[i]

    # store density
    rho = spline_W(0.0, W_table)

    # loop through neighbors to compute density
    for index in wp.hash_grid_query(boundary_grid, x, SMOOTHING_LENGTH):  # type: ignore
        distance = wp.length(x - boundary_x[index])
        if distance < SMOOTHING_LENGTH and index != i:
            rho += spline_W(distance, W_table)  # type: ignore

    boundary_phi[i] = RHO_0 / rho  # type: ignore

    # boundary_phi[tid] = FLUID_MASS


@wp.kernel
def count_same_neighbor(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    neighbor_pointer: wp.array(dtype=int),  # type: ignore
    neighbor_num: wp.array(dtype=int),  # type: ignore
    neighbor_list_index: wp.array(dtype=int),  # type: ignore
):
    tid = wp.int32(wp.tid())
    neighbor_id = wp.int32(0)

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # count neighbors
    for index in wp.hash_grid_query(grid, x, SMOOTHING_LENGTH):  # type: ignore
        distance = wp.length(x - particle_x[index])
        if distance < SMOOTHING_LENGTH and index != i:
            neighbor_id += 1

    # store number of neighbors
    neighbor_num[i] = neighbor_id
    neighbor_list_index[i] = wp.atomic_add(neighbor_pointer, 0, neighbor_id)


@wp.kernel
def store_same_neighbor(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    neighbor_list_index: wp.array(dtype=int),  # type: ignore
    neighbor_list: wp.array(dtype=int),  # type: ignore
    neighbor_distance: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())
    neighbor_id = wp.int32(0)

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    start_index = neighbor_list_index[i]

    # get local particle variables
    x = particle_x[i]

    # store neighbors
    for index in wp.hash_grid_query(grid, x, SMOOTHING_LENGTH):  # type: ignore
        distance = wp.length(x - particle_x[index])
        if distance < SMOOTHING_LENGTH and index != i:
            neighbor_list[start_index + neighbor_id] = index
            neighbor_distance[start_index + neighbor_id] = distance
            neighbor_id += 1


@wp.kernel
def count_diff_neighbor(
    grid: wp.uint64,
    neighbor_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    neighbor_x: wp.array(dtype=wp.vec3),  # type: ignore
    neighbor_pointer: wp.array(dtype=int),  # type: ignore
    neighbor_num: wp.array(dtype=int),  # type: ignore
    neighbor_list_index: wp.array(dtype=int),  # type: ignore
):
    tid = wp.int32(wp.tid())
    neighbor_id = wp.int32(0)

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # count neighbors
    for index in wp.hash_grid_query(neighbor_grid, x, SMOOTHING_LENGTH):  # type: ignore
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
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    neighbor_x: wp.array(dtype=wp.vec3),  # type: ignore
    neighbor_list_index: wp.array(dtype=int),  # type: ignore
    neighbor_list: wp.array(dtype=int),  # type: ignore
    neighbor_distance: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())
    neighbor_id = wp.int32(0)

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    start_index = neighbor_list_index[i]

    # get local particle variables
    x = particle_x[i]

    # store neighbors
    for index in wp.hash_grid_query(neighbor_grid, x, SMOOTHING_LENGTH):  # type: ignore
        distance = wp.length(x - neighbor_x[index])
        if distance < SMOOTHING_LENGTH:
            neighbor_list[start_index + neighbor_id] = index
            neighbor_distance[start_index + neighbor_id] = distance
            neighbor_id += 1


@wp.kernel
def init_pressure(
    particle_p: wp.array(dtype=float),  # type: ignore
):
    tid = wp.tid()

    # TODO check this
    particle_p[tid] = 0.0
    # particle_p[tid] /= 2.0


@wp.kernel
def compute_density(
    W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    ff_neighbor_distance: wp.array(dtype=float),  # type: ignore
    fs_neighbor_num: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    fs_neighbor_distance: wp.array(dtype=float),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
    fluid_rho: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = float(0.0)
    term_2 = float(0.0)

    # loop through neighbors to compute density
    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        term_1 += spline_W(ff_neighbor_distance[j], W_table)  # type: ignore

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        term_2 += (
            spline_W(fs_neighbor_distance[j], W_table)
            * boundary_phi[fs_neighbor_list[j]]
        )

    fluid_rho[i] = term_1 * FLUID_MASS + term_2 + spline_W(0.0, W_table) * FLUID_MASS  # type: ignore


@wp.kernel
def predict_v_adv(
    grad_W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    particle_v: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    ff_neighbor_distance: wp.array(dtype=float),  # type: ignore
    fs_neighbor_num: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    fs_neighbor_distance: wp.array(dtype=float),  # type: ignore
    fluid_rho: wp.array(dtype=float),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
    particle_v_adv: wp.array(dtype=wp.vec3),  # type: ignore
):
    tid = wp.int32(wp.tid())

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    # init terms
    term_1 = wp.vec3(0.0, 0.0, 0.0)
    term_2 = wp.vec3(0.0, 0.0, 0.0)

    x_i = particle_x[i]
    v_i = particle_v[i]

    for j in range(
        ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    ):
        index = ff_neighbor_list[j]
        x_ij = particle_x[index] - x_i
        term_1 -= (
            grad_spline_W(x_ij, grad_W_table)
            * wp.dot(v_i - particle_v[index], x_ij)
            / fluid_rho[index]
            / ff_neighbor_distance[j] ** 2.0
        )

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        x_ib = boundary_x[index] - x_i
        term_2 -= (
            grad_spline_W(x_ib, grad_W_table)
            * boundary_phi[index]
            * wp.dot(v_i, x_ib)
            / fs_neighbor_distance[j] ** 2.0
        )

    laplace_v = 10.0 * (term_1 * FLUID_MASS + term_2 / RHO_0)

    particle_v_adv[i] = (
        particle_v[i]
        + (laplace_v * VIS_MU / fluid_rho[i] + wp.vec3(0.0, -GRAVITY, 0.0)) * dt
    )


@wp.kernel
def predict_rho_adv(
    grad_W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
    particle_v_adv: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_v: wp.array(dtype=wp.vec3),  # type: ignore
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    fs_neighbor_num: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
    fluid_rho: wp.array(dtype=float),  # type: ignore
    fluid_rho_adv: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

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
        grad_W_ij = grad_spline_W(particle_x[index] - x, grad_W_table)
        term_1 += wp.dot(v_ij, grad_W_ij)

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        v_ib = v_adv - boundary_v[index]
        grad_W_ib = grad_spline_W(boundary_x[index] - x, grad_W_table)
        term_2 += wp.dot(v_ib, grad_W_ib) * boundary_phi[index]

    fluid_rho_adv[i] = fluid_rho[i] + (term_1 * FLUID_MASS + term_2) * dt


@wp.kernel
def compute_term_d(
    grad_W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    dt: float,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
    fluid_rho: wp.array(dtype=float),  # type: ignore
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    fs_neighbor_num: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
    term_d_ii: wp.array(dtype=wp.vec3),  # type: ignore
    term_d_ij: wp.array(dtype=wp.vec3),  # type: ignore
    term_d_ji: wp.array(dtype=wp.vec3),  # type: ignore
):
    tid = wp.int32(wp.tid())

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
        grad_W_ij = grad_spline_W(particle_x[index] - x, grad_W_table)
        term_1 -= grad_W_ij

        term_d_ij[j] = -grad_W_ij * term_3 / fluid_rho[index] ** 2.0  # type: ignore
        term_d_ji[j] = grad_W_ij * term_3 / rho**2.0

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        grad_W_ib = grad_spline_W(boundary_x[index] - x, grad_W_table)
        term_2 -= grad_W_ib * boundary_phi[index]

    term_d_ii[i] = (term_1 * FLUID_MASS + term_2) * dt**2.0 / rho**2.0


@wp.kernel
def compute_term_a(
    grad_W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    fs_neighbor_num: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
    term_d_ii: wp.array(dtype=wp.vec3),  # type: ignore
    term_d_ji: wp.array(dtype=wp.vec3),  # type: ignore
    term_a_ii: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

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
        grad_W_ij = grad_spline_W(particle_x[ff_neighbor_list[j]] - x, grad_W_table)
        term_1 += wp.dot((d_ii - term_d_ji[j]), grad_W_ij)

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        grad_W_ib = grad_spline_W(boundary_x[index] - x, grad_W_table)
        term_2 += wp.dot(d_ii, grad_W_ib) * boundary_phi[index]

    term_a_ii[i] = term_1 * FLUID_MASS + term_2


@wp.kernel
def compute_term_Ap_1(
    grad_W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    particle_p: wp.array(dtype=float),  # type: ignore
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    term_d_ii: wp.array(dtype=wp.vec3),  # type: ignore
    term_d_ij: wp.array(dtype=wp.vec3),  # type: ignore
    term_d_ji: wp.array(dtype=wp.vec3),  # type: ignore
    sum_d_ij_p_j: wp.array(dtype=wp.vec3),  # type: ignore
    term_Ap_i: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

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
        term_1 += wp.dot(term, grad_spline_W(particle_x[index] - x, grad_W_table))
        sum_d_ij_p_j[i] += term_d_ij[j] * particle_p[index]

    term_Ap_i[i] = term_1 * FLUID_MASS


@wp.kernel
def compute_term_Ap_2(
    grad_W_table: wp.array(dtype=wp.float32),  # type: ignore
    fluid_grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    particle_p: wp.array(dtype=float),  # type: ignore
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
    ff_neighbor_num: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list: wp.array(dtype=int),  # type: ignore
    ff_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    fs_neighbor_num: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list: wp.array(dtype=int),  # type: ignore
    fs_neighbor_list_index: wp.array(dtype=int),  # type: ignore
    boundary_phi: wp.array(dtype=float),  # type: ignore
    term_a_ii: wp.array(dtype=float),  # type: ignore
    sum_d_ij_p_j: wp.array(dtype=wp.vec3),  # type: ignore
    term_Ap_i: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

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
            sum_d_ij_p_j_i - sum_d_ij_p_j[index],
            grad_spline_W(particle_x[index] - x, grad_W_table),
        )

    for j in range(
        fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    ):
        index = fs_neighbor_list[j]
        term_2 += (
            wp.dot(sum_d_ij_p_j_i, grad_spline_W(boundary_x[index] - x, grad_W_table))
            * boundary_phi[index]
        )

    term_Ap_i[i] += term_1 * FLUID_MASS + term_2 + term_a_ii[i] * particle_p[i]


@wp.kernel
def update_p(
    fluid_grid: wp.uint64,
    fluid_rho_adv: wp.array(dtype=float),  # type: ignore
    term_a_ii: wp.array(dtype=float),  # type: ignore
    term_Ap_i: wp.array(dtype=float),  # type: ignore
    particle_p: wp.array(dtype=float),  # type: ignore
):
    tid = wp.int32(wp.tid())

    # order threads by cell
    i = wp.hash_grid_point_id(fluid_grid, tid)

    term_a_ii_i = term_a_ii[i]
    if term_a_ii_i > -INV_SMALL and term_a_ii_i < INV_SMALL:
        term_a_ii_i = wp.sign(term_a_ii_i) * INV_SMALL

    particle_p[i] += (RHO_0 - fluid_rho_adv[i] - term_Ap_i[i]) * OMEGA / term_a_ii_i
    particle_p[i] = wp.max(particle_p[i], 0.0)


@wp.kernel
def update_rho_error(
    # dt: float,
    # fluid_grid: wp.uint64,
    # particle_x: wp.array(dtype=wp.vec3),
    # particle_p: wp.array(dtype=float),
    # boundary_x: wp.array(dtype=wp.vec3),
    # ff_neighbor_num: wp.array(dtype=int),
    # ff_neighbor_list: wp.array(dtype=int),
    # ff_neighbor_list_index: wp.array(dtype=int),
    # fs_neighbor_num: wp.array(dtype=int),
    # fs_neighbor_list: wp.array(dtype=int),
    # fs_neighbor_list_index: wp.array(dtype=int),
    # boundary_phi: wp.array(dtype=float),
    # fluid_rho: wp.array(dtype=float),
    fluid_rho_adv: wp.array(dtype=float),  # type: ignore
    term_Ap_i: wp.array(dtype=float),  # type: ignore
    sum_rho_error: wp.array(dtype=float),  # type: ignore
    num_rho_error: wp.array(dtype=int),  # type: ignore
    # rho_to_check: wp.array(dtype=float),
    # delta_v_p: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    updated_rho = fluid_rho_adv[tid] + term_Ap_i[tid]
    if updated_rho > RHO_0:  # TODO check this
        wp.atomic_add(sum_rho_error, 0, wp.abs(updated_rho / RHO_0 - 1.0))
        wp.atomic_add(num_rho_error, 0, 1)

    # # order threads by cell
    # i = wp.hash_grid_point_id(fluid_grid, tid)

    # # init terms
    # term_1 = wp.vec3(0.0, 0.0, 0.0)
    # term_2 = wp.vec3(0.0, 0.0, 0.0)

    # x_i = particle_x[i]
    # p_inv_rho2_i = particle_p[i] / fluid_rho[i] ** 2.0

    # # loop through neighbors to compute density
    # for j in range(
    #     ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
    # ):
    #     index = ff_neighbor_list[j]
    #     grad_W_ij = grad_spline_W(particle_x[index] - x_i)
    #     term_1 -= (
    #         p_inv_rho2_i + particle_p[index] / fluid_rho[index] ** 2.0
    #     ) * grad_W_ij

    # for j in range(
    #     fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
    # ):
    #     index = fs_neighbor_list[j]
    #     grad_W_ib = grad_spline_W(boundary_x[index] - x_i)
    #     term_2 -= p_inv_rho2_i * grad_W_ib * boundary_phi[index]

    # delta_v_p[i] = (term_1 * FLUID_MASS + term_2) * dt
    # rho_to_check[tid] = updated_rho


# @wp.kernel
# def check_rho_error(
#     dt: float,
#     fluid_grid: wp.uint64,
#     particle_x: wp.array(dtype=wp.vec3),
#     boundary_x: wp.array(dtype=wp.vec3),
#     ff_neighbor_num: wp.array(dtype=int),
#     ff_neighbor_list: wp.array(dtype=int),
#     ff_neighbor_list_index: wp.array(dtype=int),
#     fs_neighbor_num: wp.array(dtype=int),
#     fs_neighbor_list: wp.array(dtype=int),
#     fs_neighbor_list_index: wp.array(dtype=int),
#     boundary_phi: wp.array(dtype=float),
#     fluid_rho_adv: wp.array(dtype=float),
#     rho_to_check: wp.array(dtype=float),
#     delta_v_p: wp.array(dtype=wp.vec3),
#     rho_wrong_max: wp.array(dtype=float),
# ):
#     tid = wp.int32(wp.tid())

#     # order threads by cell
#     i = wp.hash_grid_point_id(fluid_grid, tid)

#     # init terms
#     term_1 = float(0.0)
#     term_2 = float(0.0)

#     x_i = particle_x[i]
#     v_i = delta_v_p[i]

#     # loop through neighbors to compute density
#     for j in range(
#         ff_neighbor_list_index[i], ff_neighbor_list_index[i] + ff_neighbor_num[i]
#     ):
#         index = ff_neighbor_list[j]
#         grad_W_ij = grad_spline_W(particle_x[index] - x_i)
#         term_1 += wp.dot(v_i - delta_v_p[index], grad_W_ij)

#     for j in range(
#         fs_neighbor_list_index[i], fs_neighbor_list_index[i] + fs_neighbor_num[i]
#     ):
#         index = fs_neighbor_list[j]
#         grad_W_ib = grad_spline_W(boundary_x[index] - x_i)
#         term_2 += wp.dot(v_i, grad_W_ib) * boundary_phi[index]

#     rho_check = fluid_rho_adv[i] + (term_1 * FLUID_MASS + term_2) * dt
#     rho_wrong = wp.abs(rho_check - rho_to_check[i])
#     wp.atomic_max(rho_wrong_max, 0, rho_wrong)


@wp.kernel
def set_rigid_v(
    boundary_hide_n: int,
    rigid_v: wp.vec3,
    boundary_v: wp.array(dtype=wp.vec3),  # type: ignore
):
    tid = wp.tid()
    if tid >= boundary_hide_n:
        boundary_v[tid] = rigid_v


@wp.kernel
def update_rigid_x(
    dt: float,
    boundary_hide_n: int,
    boundary_v: wp.array(dtype=wp.vec3),  # type: ignore
    boundary_x: wp.array(dtype=wp.vec3),  # type: ignore
):
    tid = wp.tid()
    if tid >= boundary_hide_n:
        boundary_x[tid] += boundary_v[tid] * dt


@wp.kernel
def kick(
    inv_dt: float,
    particle_p: wp.array(dtype=float),  # type: ignore
    particle_v_adv: wp.array(dtype=wp.vec3),  # type: ignore
    term_d_ii: wp.array(dtype=wp.vec3),  # type: ignore
    sum_d_ij_p_j: wp.array(dtype=wp.vec3),  # type: ignore
    particle_v: wp.array(dtype=wp.vec3),  # type: ignore
    particle_v_max: wp.array(dtype=float),  # type: ignore
):
    tid = wp.tid()
    delta_v = inv_dt * (term_d_ii[tid] * particle_p[tid] + sum_d_ij_p_j[tid])
    v = particle_v_adv[tid] + delta_v
    particle_v[tid] = v
    wp.atomic_max(particle_v_max, 0, wp.length(v))


@wp.kernel
def drift(
    dt: float,
    particle_v: wp.array(dtype=wp.vec3),  # type: ignore
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    # penetration_times: wp.array(dtype=int),
):
    tid = wp.tid()
    particle_x[tid] += particle_v[tid] * dt
    # new_pos = particle_x[tid] + particle_v[tid] * dt

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

    # particle_x[tid] = new_pos
    # if penetration_detected:
    #     wp.atomic_add(penetration_times, 0, 1)


class IISPH:

    def __init__(self, stage_path="example_sph.usd", preview=False, verbose=False):
        self.verbose = verbose

        # render params
        self.frame_dt = 1.0 / FPS
        self.sim_time = 0.0
        self.last_frame_time = 0.0

        # simulation params
        self.dt = TIME_STEP_MAX
        self.inv_dt = 1 / self.dt
        self.boundary_layer = 3

        if scene_type == SceneType.HOUSE:
            min_point = (BOX_WIDTH * 0.7, BOX_HEIGHT * 0.01, BOX_LENGTH * 0.8)
            max_point = (BOX_WIDTH * 0.99, BOX_HEIGHT * 0.5, BOX_LENGTH * 0.99)
            self.init_particles(min_point, max_point)
        elif scene_type == SceneType.PLANE:
            min_point = (BOX_WIDTH * 0.45, BOX_HEIGHT * 0.05, BOX_LENGTH * 0.05)
            max_point = (BOX_WIDTH * 0.55, BOX_HEIGHT * 0.95, BOX_LENGTH * 0.95)
            self.init_particles(min_point, max_point, boundary_layers=0)
        elif scene_type == SceneType.HAND:
            min_point = (BOX_WIDTH * 0.48, BOX_HEIGHT * 0.1, BOX_LENGTH * 0.495)
            max_point = (BOX_WIDTH * 0.52, BOX_HEIGHT * 0.99, BOX_LENGTH * 0.505)
            self.init_particles(min_point, max_point)
        else:
            raise NotImplementedError("Not implemented scene type")

        # create hash array
        self.fluid_grid = wp.HashGrid(
            int(BOX_WIDTH / SMOOTHING_LENGTH / 4),
            int(BOX_HEIGHT / SMOOTHING_LENGTH / 4),
            int(BOX_LENGTH / SMOOTHING_LENGTH / 4),
        )
        self.boundary_grid = wp.HashGrid(
            int(BOX_WIDTH / SMOOTHING_LENGTH / 4),
            int(BOX_HEIGHT / SMOOTHING_LENGTH / 4),
            int(BOX_LENGTH / SMOOTHING_LENGTH / 4),
        )

        # allocate arrays
        self.v = wp.zeros(self.n, dtype=wp.vec3)  # type: ignore
        self.v_adv = wp.zeros(self.n, dtype=wp.vec3)  # type: ignore
        self.rho = wp.zeros(self.n, dtype=float)  # type: ignore
        self.rho_adv = wp.zeros(self.n, dtype=float)  # type: ignore
        self.a = wp.zeros(self.n, dtype=wp.vec3)  # type: ignore
        self.p = wp.zeros(self.n, dtype=float)  # type: ignore
        self.boundary_phi = wp.zeros(self.boundary_n, dtype=float)  # type: ignore
        self.sum_rho_error = wp.zeros(1, dtype=float)  # type: ignore
        self.num_rho_error = wp.zeros(1, dtype=int)  # type: ignore
        self.term_a_ii = wp.zeros(self.n, dtype=float)  # type: ignore
        self.term_d_ii = wp.zeros(self.n, dtype=wp.vec3)  # type: ignore
        self.term_d_ij = wp.zeros(self.n * 60, dtype=wp.vec3)  # type: ignore
        self.term_d_ji = wp.zeros(self.n * 60, dtype=wp.vec3)  # type: ignore
        self.term_Ap_i = wp.zeros(self.n, dtype=float)  # type: ignore
        self.sum_d_ij_p_j = wp.zeros(self.n, dtype=wp.vec3)  # type: ignore
        self.ff_neighbor_num = wp.zeros(self.n, dtype=int)  # type: ignore
        self.ff_neighbor_list = wp.zeros(self.n * 60, dtype=int)  # type: ignore
        self.ff_neighbor_distance = wp.zeros(self.n * 60, dtype=float)  # type: ignore
        self.ff_neighbor_list_index = wp.zeros(self.n, dtype=int)  # type: ignore
        self.fs_neighbor_num = wp.zeros(self.n, dtype=int)  # type: ignore
        self.fs_neighbor_list = wp.zeros(self.n * 60, dtype=int)  # type: ignore
        self.fs_neighbor_distance = wp.zeros(self.n * 60, dtype=float)  # type: ignore
        self.fs_neighbor_list_index = wp.zeros(self.n, dtype=int)  # type: ignore
        # self.penetration_times = wp.zeros(1, dtype=int)
        # self.delta_v_p = wp.zeros(self.n, dtype=wp.vec3)
        # self.rho_to_check = wp.zeros(self.n, dtype=float)

        # compute PHI value of boundary particles
        self.boundary_grid.build(self.boundary_x, SMOOTHING_LENGTH)
        wp.launch(
            kernel=compute_boundary_density,
            dim=self.boundary_n,
            inputs=[
                W_table,
                self.boundary_grid.id,
                self.boundary_x,
            ],
            outputs=[self.boundary_phi],
        )

        # renderer
        if stage_path:
            self.renderer = warp.render.UsdRenderer(stage_path, fps=FPS)
        else:
            self.renderer = None

        if preview:
            previewer = warp.render.OpenGLRenderer(
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

    def init_particles(
        self,
        min_point=None,
        max_point=None,
        spacing=DIAMETER,
        fluid_depth=0.0,
        boundary_layers=3,
    ):
        """
        1. Generate boundary particles for the box
        2. Generate fluid particles in a rectangular region if min_point and max_point specified
        3. Fill the bottom of the box with fluid up to fluid_depth height
        """
        assert spacing > 0, "Particle spacing must be positive"

        x_range = range(-boundary_layers, int(BOX_WIDTH / spacing) + boundary_layers)
        y_range = range(-boundary_layers, int(BOX_HEIGHT / spacing) + boundary_layers)
        z_range = range(-boundary_layers, int(BOX_LENGTH / spacing) + boundary_layers)

        x_min, x_max = 0, int(BOX_WIDTH / spacing) - 1
        y_min, y_max = 0, int(BOX_HEIGHT / spacing) - 1
        z_min, z_max = 0, int(BOX_LENGTH / spacing) - 1

        boundary_particles = []
        fluid_particles = []

        fluid_y_max = int(fluid_depth / spacing) if fluid_depth > 0 else -1

        for x_idx in x_range:
            for y_idx in y_range:
                for z_idx in z_range:
                    x = x_idx * spacing
                    y = y_idx * spacing
                    z = z_idx * spacing

                    is_outside = (
                        x_idx < x_min
                        or x_idx > x_max
                        or y_idx < y_min
                        or y_idx > y_max
                        or z_idx < z_min
                        or z_idx > z_max
                    )

                    is_within_layers = (
                        x_idx >= x_min - boundary_layers
                        and x_idx <= x_max + boundary_layers
                        and y_idx >= y_min - boundary_layers
                        and y_idx <= y_max + boundary_layers
                        and z_idx >= z_min - boundary_layers
                        and z_idx <= z_max + boundary_layers
                    )

                    # Generate boundary particles
                    if is_outside and is_within_layers:
                        boundary_particles.append([x, y, z])

                    # Generate fluid pool particles
                    elif (not is_outside) and fluid_depth > 0 and y_idx <= fluid_y_max:
                        in_rect_region = False
                        if min_point is not None and max_point is not None:
                            in_rect_region = (
                                min_point[0] - spacing < x < max_point[0] + spacing
                                and min_point[1] - spacing < y < max_point[1] + spacing
                                and min_point[2] - spacing < z < max_point[2] + spacing
                            )

                        if not in_rect_region:
                            fluid_particles.append([x, y, z])

        self.boundary_hide_n = len(boundary_particles)

        # Add fluid particles in rectangular region if specified
        if min_point is not None and max_point is not None:
            assert (
                0 < min_point[0] < max_point[0] < BOX_WIDTH
            ), "Invalid X range for rectangle"
            assert (
                0 < min_point[1] < max_point[1] < BOX_HEIGHT
            ), "Invalid Y range for rectangle"
            assert (
                0 < min_point[2] < max_point[2] < BOX_LENGTH
            ), "Invalid Z range for rectangle"

            x_rect = np.arange(min_point[0], max_point[0], spacing)
            y_rect = np.arange(min_point[1], max_point[1], spacing)
            z_rect = np.arange(min_point[2], max_point[2], spacing)

            xx, yy, zz = np.meshgrid(x_rect, y_rect, z_rect, indexing="ij")
            rect_positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()]).T

            for pos in rect_positions:
                fluid_particles.append(pos)

        # Add rigid particles from mesh
        model_list = []

        def add_model(
            filename,
            scale=1.0,
            pos: np.ndarray = np.zeros(3),
            rot=np.eye(3),
            spacing=DIAMETER,
        ):
            model = load_model(filename, scale, pos, rot, spacing)
            print(f"Loaded model {filename} with {len(model)} particles")
            model_list.append(model)

        if scene_type == SceneType.HOUSE:
            add_model("house.obj", 2e-2, np.array([3.0, 0.0, 4.0]))
        elif scene_type == SceneType.PLANE:
            pos = np.array([0.0, 4.0, 4.0])
            rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            add_model("plane.obj", 3e-3, pos, rot)
        elif scene_type == SceneType.HAND:
            pos = np.array([4.5, 2.0, 1.0])
            angle = np.radians(160)
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            add_model("hand.obj", 0.3, pos, rot)
        else:
            raise NotImplementedError("Not implemented scene type")

        # Convert to warp arrays and return
        for model in model_list:
            boundary_particles.extend(model.tolist())
        self.boundary_x = wp.array(boundary_particles, dtype=wp.vec3)
        self.boundary_v = wp.zeros_like(self.boundary_x)
        self.boundary_n = len(boundary_particles)
        self.x = wp.array(fluid_particles, dtype=wp.vec3)
        self.n = len(fluid_particles)
        print(f"Initialized {self.n} fluid and {self.boundary_n} boundary particles")

        # TODO remove this
        idx = np.arange(self.boundary_hide_n)
        mask = np.random.rand(self.boundary_hide_n) < 1e-2
        self.select_boundary = self.boundary_x.numpy()[idx[mask]]

    def step(self):  # TODO use CUDA graph capture
        with wp.ScopedTimer("step", active=self.verbose):
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
                        W_table,
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
                        grad_W_table,
                        self.fluid_grid.id,
                        self.dt,
                        self.x,
                        self.v,
                        self.boundary_x,
                        self.ff_neighbor_num,
                        self.ff_neighbor_list,
                        self.ff_neighbor_list_index,
                        self.ff_neighbor_distance,
                        self.fs_neighbor_num,
                        self.fs_neighbor_list,
                        self.fs_neighbor_list_index,
                        self.fs_neighbor_distance,
                        self.rho,
                        self.boundary_phi,
                    ],
                    outputs=[self.v_adv],
                )

                # predict density advection
                if self.sim_time == 0.0:
                    if scene_type == SceneType.PLANE:
                        wp.launch(
                            kernel=set_rigid_v,
                            dim=self.boundary_n,
                            inputs=[
                                self.boundary_hide_n,
                                wp.vec3(2.0, 0.0, 0.0),
                            ],
                            outputs=[self.boundary_v],
                        )
                    elif scene_type == SceneType.HAND:
                        wp.launch(
                            kernel=set_rigid_v,
                            dim=self.boundary_n,
                            inputs=[
                                self.boundary_hide_n,
                                wp.vec3(0.0, 0.1, 0.0),
                            ],
                            outputs=[self.boundary_v],
                        )

                wp.launch(
                    kernel=predict_rho_adv,
                    dim=self.n,
                    inputs=[
                        grad_W_table,
                        self.fluid_grid.id,
                        self.dt,
                        self.x,
                        self.boundary_x,
                        self.v_adv,
                        self.boundary_v,
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
                        grad_W_table,
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
                        grad_W_table,
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
                wp.launch(
                    kernel=compute_term_Ap_1,
                    dim=self.n,
                    inputs=[
                        grad_W_table,
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
                        grad_W_table,
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

                loop = 0
                while (loop < 2) or (self.average_rho_error > ETA):
                    if loop > 100:
                        # self.raise_error("Pressure solver did not converge.")
                        print(
                            f"Pressure solver did not converge at {self.sim_time:.2f}s"
                        )
                        break

                    wp.launch(
                        kernel=update_p,
                        dim=self.n,
                        inputs=[
                            self.fluid_grid.id,
                            self.rho_adv,
                            self.term_a_ii,
                            self.term_Ap_i,
                        ],
                        outputs=[self.p],
                    )

                    wp.launch(
                        kernel=compute_term_Ap_1,
                        dim=self.n,
                        inputs=[
                            grad_W_table,
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
                            grad_W_table,
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

                    self.sum_rho_error = wp.zeros(1, dtype=float)  # type: ignore
                    self.num_rho_error = wp.zeros(1, dtype=int)  # type: ignore
                    wp.launch(
                        kernel=update_rho_error,
                        dim=self.n,
                        inputs=[
                            # self.dt,
                            # self.fluid_grid.id,
                            # self.x,
                            # self.p,
                            # self.boundary_x,
                            # self.ff_neighbor_num,
                            # self.ff_neighbor_list,
                            # self.ff_neighbor_list_index,
                            # self.fs_neighbor_num,
                            # self.fs_neighbor_list,
                            # self.fs_neighbor_list_index,
                            # self.boundary_phi,
                            # self.rho,
                            self.rho_adv,
                            self.term_Ap_i,
                        ],
                        outputs=[
                            self.sum_rho_error,
                            self.num_rho_error,
                            # self.rho_to_check,
                            # self.delta_v_p,
                        ],
                    )

                    loop += 1

            with wp.ScopedTimer("integration", active=self.verbose):
                v_max = wp.zeros(1, dtype=float)  # type: ignore
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
                    ],
                )

                # rho_wrong_max = wp.zeros(1, dtype=float)
                # wp.launch(
                #     kernel=check_rho_error,
                #     dim=self.n,
                #     inputs=[
                #         self.dt,
                #         self.fluid_grid.id,
                #         self.x,
                #         self.boundary_x,
                #         self.ff_neighbor_num,
                #         self.ff_neighbor_list,
                #         self.ff_neighbor_list_index,
                #         self.fs_neighbor_num,
                #         self.fs_neighbor_list,
                #         self.fs_neighbor_list_index,
                #         self.boundary_phi,
                #         self.rho_adv,
                #         self.rho_to_check,
                #         self.delta_v_p,
                #     ],
                #     outputs=[rho_wrong_max],
                # )

                # rho_wrong = rho_wrong_max.numpy()[0]
                # if rho_wrong > 9e-5:
                #     self.previewer.paused = True
                #     print(f"Rho wrong: {rho_wrong:.6f}")

                # drift
                wp.launch(
                    kernel=drift,
                    dim=self.n,
                    inputs=[self.dt, self.v],
                    outputs=[
                        self.x,
                        # self.penetration_times,
                    ],
                )

                wp.launch(
                    kernel=update_rigid_x,
                    dim=self.boundary_n,
                    inputs=[
                        self.dt,
                        self.boundary_hide_n,
                        self.boundary_v,
                    ],
                    outputs=[self.boundary_x],
                )

            self.sim_time += self.dt
            self.dt = wp.min(  # CFL condition
                0.4 * DIAMETER / max(v_max.numpy()[0], INV_SMALL), TIME_STEP_MAX
            )
            self.inv_dt = 1 / self.dt

    def activate_renderer(self, renderer):
        renderer.begin_frame(self.sim_time)
        renderer.render_points(
            points=self.x.numpy(),
            radius=wp.constant(DIAMETER / 1.6),
            name="fluid",
            colors=(0.5, 0.5, 0.8),
        )
        rigid_body = self.boundary_x.numpy()[self.boundary_hide_n :]
        renderer.render_points(
            points=rigid_body,
            radius=wp.constant(DIAMETER / 1.4),
            name="boundary",
            colors=(0.6, 0.7, 0.8),
        )
        # renderer.render_points(
        #     points=self.select_boundary,
        #     radius=wp.constant(DIAMETER / 3.0),
        #     name="box",
        #     colors=(0.8, 0.6, 0.6),
        # )
        renderer.end_frame()

    def render(self):
        if self.previewer:
            self.activate_renderer(self.previewer)

        if self.renderer is None:
            return

        if self.sim_time - self.last_frame_time >= self.frame_dt:
            with wp.ScopedTimer("render", active=self.verbose):
                self.activate_renderer(self.renderer)
            self.last_frame_time = self.sim_time

    def neighbor_search(self):
        """
        Neighbor search for both fluid-fluid and fluid-boundary interactions.
        """

        # build grid of fluid particles
        self.fluid_grid.build(self.x, SMOOTHING_LENGTH)
        if DYNAMIC_SCENE:
            self.boundary_grid.build(self.boundary_x, SMOOTHING_LENGTH)

        # search fluid neighbors for fluid
        neighbor_pointer = wp.zeros(1, dtype=int)  # type: ignore

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

        wp.launch(
            kernel=store_same_neighbor,
            dim=self.n,
            inputs=[
                self.fluid_grid.id,
                self.x,
                self.ff_neighbor_list_index,
            ],
            outputs=[
                self.ff_neighbor_list,
                self.ff_neighbor_distance,
            ],
        )

        # search boundary neighbors for fluid
        neighbor_pointer = wp.zeros(1, dtype=int)  # type: ignore

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

        wp.launch(
            kernel=store_diff_neighbor,
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
    def average_rho_error(self):
        num_rho_error = self.num_rho_error.numpy()[0]
        if num_rho_error == 0:
            return 0.0
        return self.sum_rho_error.numpy()[0] / num_rho_error

    @property
    def window_closed(self):
        if self.previewer:
            return self.previewer.has_exit
        else:
            return False
