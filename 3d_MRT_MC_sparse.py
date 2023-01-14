import taichi as ti
import numpy as np
import math
from sympy import inverse_mellin_transform
from pyevtk.hl import gridToVTK
import pandas as pd
import vtk
import time

ti.init(arch=ti.cpu, dynamic_index=False)
# ti.init(arch=ti.cpu,cpu_max_num_threads=32, dynamic_index=False)

# LBM parameters
Q = 19
cs2 = 1.0 / 3.0

"""Definition of LBM weights"""
t0 = 1.0 / 3.0
t1 = 1.0 / 18.0
t2 = 1.0 / 36.0
# t = np.array([t0, t1, t1, t1, t2, t2, t2, t2, t2, t2, t1, t1, t1, t2, t2, t2, t2, t2, t2])
t = np.array(
    [t0, t1, t1, t1, t1, t1, t1, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2]
)

"""Definition of Shan-Chen factors for force computation"""
w0 = 0
w1 = 2
w2 = 1
w = np.array(
    [t0, t1, t1, t1, t1, t1, t1, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2, t2]
)


# x component of predefined velocity in Q directions
e_xyz_list = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
]

# Predefined compound types
i32_vec3d = ti.types.vector(3, ti.i32)
f32_vec3d = ti.types.vector(3, ti.f32)
f32_vec2d = ti.types.vector(2, ti.f32)

G_ls = -0.4
G_lo = 4.0
G_gs = -G_ls
vtk_dstep = 500

# MRT operator
niu_l = 0.2  # 0.01
niu_g = 0.2  # 0.00018

tau_lf = 3.0 * niu_l + 0.5
tau_gf = 3.0 * niu_g + 0.5

# Diagonal relaxation matrix for water
s_lv = 1.0 / tau_lf
s_lother = 8.0 * (2.0 - s_lv) / (8.0 - s_lv)
S_dig_np_l = np.array(
    [
        1,
        s_lv,
        s_lv,
        1,
        s_lother,
        1,
        s_lother,
        1,
        s_lother,
        s_lv,
        s_lv,
        s_lv,
        s_lv,
        s_lv,
        s_lv,
        s_lv,
        s_lother,
        s_lother,
        s_lother,
    ]
)

# Diagonal relaxation matrix for oil
s_gv = 1.0 / tau_gf
s_gother = 8.0 * (2.0 - s_gv) / (8.0 - s_gv)
S_dig_np_g = np.array(
    [
        1,
        s_gv,
        s_gv,
        1,
        s_gother,
        1,
        s_gother,
        1,
        s_gother,
        s_gv,
        s_gv,
        s_gv,
        s_gv,
        s_gv,
        s_gv,
        s_gv,
        s_gother,
        s_gother,
        s_gother,
    ]
)

M_np = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -2, -2, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, -2, 2, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, -2, 2, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
        [0, 2, 2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
        [0, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
        [0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1],
    ]
)

inv_M_np = np.linalg.inv(M_np)
reversed_e = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
)

# Input paramters
lx = 100
ly = lz = 100
x = np.linspace(0, lx, lx)
y = np.linspace(0, ly, ly)
z = np.linspace(0, lz, lz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

IniPerturbRate = 1
carn_star = False
T_Tc = 0.7

rhos = 3
rho_g = 2
rho_l = 2

rho_i = 2

solid_np = np.zeros((lx, ly, lz), dtype=np.int8)
solid_count = 0

for x in range(lx):
    for y in range(ly):
        for z in range(lz):
            if z == 0 or z == lz - 1:
                solid_np[x, y, z] = 1
                solid_count += 1


def place_sphere(x, y, z, R):
    global solid_count

    xmin = x - R
    ymin = y - R
    zmin = z - R

    xmax = x + R
    ymax = y + R
    zmax = z + R

    for px in range(xmin, xmax + 1):
        for py in range(ymin, ymax + 1):
            for pz in range(zmin, zmax + 1):
                dx = px - x
                dy = py - y
                dz = pz - z
                dist2 = dx * dx + dy * dy + dz * dz
                R2 = R * R
                if dist2 < R2:
                    near_px = (
                        math.floor(px + 0.5)
                        if math.floor(px + 0.5)
                        else math.floor(px + 0.5) + lx
                    )
                    near_py = (
                        math.floor(py + 0.5)
                        if math.floor(py + 0.5)
                        else math.floor(py + 0.5) + ly
                    )
                    near_pz = (
                        math.floor(pz + 0.5)
                        if math.floor(pz + 0.5)
                        else math.floor(pz + 0.5) + lz
                    )

                    if near_px >= lx:
                        near_px -= lx
                    if near_py >= ly:
                        near_py -= ly
                    if near_pz >= lz:
                        near_pz -= lz

                    solid_np[near_px, near_py, near_pz] = 1
                    solid_count += 1


def read_positions(position_filename):
    global solid_count
    i = 0
    with open(position_filename) as f:
        Lines = f.readlines()
    for line in Lines:
        i += 1
        k = float(line)
        k = int(k)
        if i == 1:
            x = k
        elif i == 2:
            y = k
        elif i == 3:
            z = k
        else:
            i = 0
            r = k
            solid_count += 1
            place_sphere(x, y, z, r)


# grain_diameter = 20
# read_positions("./lx" + str(lx) + "_" + str(int(grain_diameter)))
# print(
#     "The computational domain has {} grains with {} lu in diameter.".format(
#         solid_count, grain_diameter
#     ),
# )
# positions_767balls_2.5-7.5radius_0.5porosity_res100
place_sphere(50, 50, 50, 10)


@ti.data_oriented
class lbm_single_phase:
    def __init__(self, filename="Stats.txt", sparse_mem=True):

        self.nb_solid_nodes = ti.field(ti.i32, shape=())
        self.step = ti.field(ti.i32, shape=())

        self.step[None] = 0
        self.IS_SOLID = ti.field(ti.i32)

        self.collide_f_l = ti.Vector.field(Q, ti.f32)
        self.stream_f_l = ti.Vector.field(Q, ti.f32)
        self.rho_l = ti.field(ti.f32)
        self.v_l = ti.Vector.field(3, ti.f32)
        self.force_l = ti.Vector.field(3, ti.f32)
        self.psi_l = ti.field(ti.f32)

        self.collide_f_g = ti.Vector.field(Q, ti.f32)
        self.stream_f_g = ti.Vector.field(Q, ti.f32)
        self.rho_g = ti.field(ti.f32)
        self.v_g = ti.Vector.field(3, ti.f32)
        self.force_g = ti.Vector.field(3, ti.f32)
        self.psi_g = ti.field(ti.f32)

        self.pressure = ti.field(ti.f32)

        if sparse_mem == False:
            ti.root.dense(ti.ijk, (lx, ly, lz)).place(
                self.force_l,
                self.psi_l,
                self.rho_l,
                self.v_l,
                self.force_g,
                self.psi_g,
                self.rho_g,
                self.v_g,
                self.IS_SOLID,
                self.collide_f_l,
                self.collide_f_g,
                self.stream_f_l,
                self.stream_f_g,
                self.pressure,
            )

        else:
            n_mem_partition = 3  # Generate blocks of 3X3x3
            ti.root.dense(ti.ijk, (lx, ly, lz)).place(
                self.IS_SOLID,
                self.psi_l,
                self.rho_l,
                self.psi_g,
                self.rho_g,
            )
            cell = ti.root.pointer(
                ti.ijk,
                (
                    lx // n_mem_partition + 1,
                    ly // n_mem_partition + 1,
                    lz // n_mem_partition + 1,
                ),
            )
            cell.dense(
                ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)
            ).place(
                self.force_l,
                self.v_l,
                self.force_g,
                self.v_g,
                self.collide_f_l,
                self.collide_f_g,
                self.stream_f_l,
                self.stream_f_g,
                self.pressure,
            )

        # to compare if disassembled for-loop is faster or not
        self.M = ti.field(ti.f32, shape=(Q, Q))
        self.inv_M = ti.field(ti.f32, shape=(Q, Q))

        self.M.from_numpy(M_np)
        self.inv_M.from_numpy(inv_M_np)

        self.w = ti.field(ti.f32, shape=(Q))
        self.w.from_numpy(w)

        self.M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())
        self.inv_M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())

        self.M_mat[None] = ti.Matrix(M_np)
        self.inv_M_mat[None] = ti.Matrix(inv_M_np)

        self.S_dig_vec_l = ti.Vector.field(Q, ti.f32, shape=())
        self.S_dig_vec_l.from_numpy(S_dig_np_l)

        self.S_dig_vec_g = ti.Vector.field(Q, ti.f32, shape=())
        self.S_dig_vec_g.from_numpy(S_dig_np_g)

        self.e_xyz = ti.Vector.field(3, dtype=ti.i32, shape=(Q))
        self.e_xyz.from_numpy(np.array(e_xyz_list))

        self.ef_xyz = ti.Vector.field(3, dtype=ti.f32, shape=(Q))
        self.ef_xyz.from_numpy(np.array(e_xyz_list))

        # REVERSED_E stores the index of the opposite component to every component in e_xyz_np
        # For example, for [1,0,0], the opposite component is [-1,0,0] which has the index of 2 in e_xyz
        self.REVERSED_E = [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
            17,
        ]

        ti.static(self.REVERSED_E)
        self.IS_SOLID.from_numpy(solid_np)
        self.Gl = ti.field(ti.f32, shape=(2))
        self.Go = ti.field(ti.f32, shape=(2))

        self.Gl[0] = G_lo
        self.Gl[1] = G_ls

        self.Go[0] = G_lo
        self.Go[1] = G_gs

        ti.static(self.inv_M)
        ti.static(self.IS_SOLID)
        ti.static(self.S_dig_vec_g)
        ti.static(self.S_dig_vec_l)
        ti.static(self.M)
        ti.static(self.e_xyz)
        ti.static(self.ef_xyz)
        ti.static(self.w)

        ti.static(self.Gl)
        ti.static(self.Go)

    @ti.func
    def Psi(self, rho_value):
        return 1.0 - ti.exp(-rho_value)

    @ti.kernel
    def update_psi(self):
        for I in ti.grouped(self.IS_SOLID):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                self.psi_l[I] = self.Psi(self.rho_l[I])
                self.psi_g[I] = self.Psi(self.rho_g[I])
                # Kruger's
                self.pressure[I] = (
                    cs2 * (self.rho_l[I] + self.rho_g[I])
                    + cs2 * G_lo / 2 * self.psi_l[I] * self.psi_g[I]
                )

    @ti.kernel
    def update_force(self):
        for I in ti.grouped(self.IS_SOLID):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                force_l = ti.Vector([0.0, 0.0, 0.0])
                force_g = ti.Vector([0.0, 0.0, 0.0])

                for i in ti.static(range(3)):
                    for s in ti.static(range(1, Q)):
                        neighbor_pos = self.periodic_index(I + self.e_xyz[s])
                        force_l[i] += (
                            -self.psi_g[neighbor_pos]
                            * self.psi_l[I]
                            * self.w[s]
                            * self.Gl[(self.IS_SOLID[neighbor_pos])]
                            * self.ef_xyz[s][i]
                        )

                        force_g[i] += (
                            -self.psi_l[neighbor_pos]
                            * self.psi_g[I]
                            * self.w[s]
                            * self.Go[(self.IS_SOLID[neighbor_pos])]
                            * self.ef_xyz[s][i]
                        )

                self.force_l[I] = force_l
                self.force_g[I] = force_g

    @ti.kernel
    def update_velocity(self):
        for I in ti.grouped(self.collide_f_l):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:

                for i in ti.static(range(3)):
                    temp_vel_l = 0.0
                    temp_vel_g = 0.0
                    for s in ti.static(range(Q)):
                        temp_vel_l += (
                            self.collide_f_l[I][s] * self.ef_xyz[s][i] / tau_lf
                        )
                        temp_vel_g += (
                            self.collide_f_g[I][s] * self.ef_xyz[s][i] / tau_gf
                        )

                    common_vel = (temp_vel_l + temp_vel_g) / (
                        self.rho_l[I] / tau_lf + self.rho_g[I] / tau_gf
                    )

                    if self.rho_l[I] > 0.0:
                        self.v_l[I][i] = (
                            common_vel + self.force_l[I][i] * tau_lf / self.rho_l[I]
                        )

                    if self.rho_g[I] > 0.0:
                        self.v_g[I][i] = (
                            common_vel + self.force_g[I][i] * tau_gf / self.rho_g[I]
                        )

    @ti.func
    def meq_vec(self, rho_local, u):
        out = ti.Vector(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        # The D3QQ Gram-Schmidt equilibrium moments
        out[0] = 1
        out[3] = u[0]
        out[5] = u[1]
        out[7] = u[2]
        out[1] = u.dot(u)
        out[9] = 2 * u.x * u.x - u.y * u.y - u.z * u.z
        out[11] = u.y * u.y - u.z * u.z
        out[13] = u.x * u.y
        out[14] = u.y * u.z
        out[15] = u.x * u.z
        return out * rho_local

    @ti.func
    def place_fluid_sphere(self, x, y, z, R):

        xmin = x - R
        ymin = y - R
        zmin = z - R

        xmax = x + R
        ymax = y + R
        zmax = z + R

        nb_nodes = 0
        for px in range(xmin, xmax + 1):
            for py in range(ymin, ymax + 1):
                for pz in range(zmin, zmax + 1):
                    dx = px - x
                    dy = py - y
                    dz = pz - z
                    dist2 = dx * dx + dy * dy + dz * dz
                    R2 = R * R
                    if dist2 < R2:
                        near_px = ti.floor(px + 0.5, ti.i32)
                        near_py = ti.floor(py + 0.5, ti.i32)
                        near_pz = ti.floor(pz + 0.5, ti.i32)

                        if ti.floor(px + 0.5) <= 0:
                            near_px = ti.floor(px + 0.5, ti.i32) + lx
                        if ti.floor(py + 0.5) <= 0:
                            near_py = ti.floor(py + 0.5, ti.i32) + ly
                        if ti.floor(pz + 0.5) <= 0:
                            near_pz = ti.floor(pz + 0.5, ti.i32) + lz

                        if near_px >= lx:
                            near_px -= lx
                        if near_py >= ly:
                            near_py -= ly
                        if near_pz >= lz:
                            near_pz -= lz

                        if self.IS_SOLID[near_px, near_py, near_pz] == 0:
                            self.rho_l[near_px, near_py, near_pz] = 0.0
                            self.rho_g[near_px, near_py, near_pz] = rho_g
                            nb_nodes += 1
        return nb_nodes

    @ti.kernel
    def init_field(self):
        for x, y, z in self.IS_SOLID:
            self.rho_l[x, y, z] = rhos
            self.rho_g[x, y, z] = rhos
            self.psi_l[x, y, z] = self.Psi(rhos)
            self.psi_g[x, y, z] = self.Psi(rhos)

            if self.IS_SOLID[x, y, z] == 0:
                self.rho_l[x, y, z] = rho_l
                self.rho_g[x, y, z] = 0.0

        nb_nodes = self.place_fluid_sphere(50, 50, 50, 40)
        print(nb_nodes)

        for x, y, z in self.IS_SOLID:
            if self.IS_SOLID[x, y, z] == 0:
                self.psi_l[x, y, z] = self.Psi(self.rho_l[x, y, z])
                self.psi_g[x, y, z] = self.Psi(self.rho_g[x, y, z])

                for q in ti.static(range(Q)):
                    self.collide_f_l[x, y, z][q] = t[q] * self.rho_l[x, y, z]
                    self.stream_f_l[x, y, z][q] = t[q] * self.rho_l[x, y, z]

                    self.collide_f_g[x, y, z][q] = t[q] * self.rho_g[x, y, z]
                    self.stream_f_g[x, y, z][q] = t[q] * self.rho_g[x, y, z]

    # check if sparse storage works!
    @ti.kernel
    def activity_checking(self) -> int:
        nb_activated_nodes = 0
        for x, y, z in self.collide_f_g:
            nb_activated_nodes += 1
        return nb_activated_nodes

    @ti.kernel
    def collision(self):
        for I in ti.grouped(self.collide_f_l):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:

                # Matrix dot product
                m_l = self.M_mat[None] @ self.collide_f_l[I]
                m_eq_l = self.meq_vec(self.rho_l[I], self.v_l[I])
                m_l -= self.S_dig_vec_l[None] * (m_l - m_eq_l)

                m_g = self.M_mat[None] @ self.collide_f_g[I]
                m_eq_g = self.meq_vec(self.rho_g[I], self.v_g[I])
                m_g -= self.S_dig_vec_g[None] * (m_g - m_eq_g)

                self.collide_f_l[I] = ti.Vector(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                self.collide_f_l[I] += self.inv_M_mat[None] @ m_l

                self.collide_f_g[I] = ti.Vector(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                self.collide_f_g[I] += self.inv_M_mat[None] @ m_g

    @ti.kernel
    def post_collsion(self):
        """Calculate force and velocity"""
        for I in ti.grouped(self.collide_f_l):
            if I.x < lx and I.y < ly and I.z < lz and self.IS_SOLID[I] == 0:
                self.collide_f_l[I] = self.stream_f_l[I]
                self.rho_l[I] = self.collide_f_l[I].sum()

                self.collide_f_g[I] = self.stream_f_g[I]
                self.rho_g[I] = self.collide_f_g[I].sum()

    @ti.func
    def periodic_index(self, i):
        iout = i
        if i[0] < 0:
            iout[0] = lx - 1
        if i[0] > lx - 1:
            iout[0] = 0
        if i[1] < 0:
            iout[1] = ly - 1
        if i[1] > ly - 1:
            iout[1] = 0
        if i[2] < 0:
            iout[2] = lz - 1
        if i[2] > lz - 1:
            iout[2] = 0

        return iout

    @ti.kernel
    def streaming(self):
        for i in ti.grouped(self.collide_f_l):
            if self.IS_SOLID[i] == 0 and i.x < lx and i.y < ly and i.z < lz:
                for s in ti.static(range(Q)):
                    ip = self.periodic_index(i + self.e_xyz[s])
                    if self.IS_SOLID[ip] == 0:
                        self.stream_f_l[ip][s] = self.collide_f_l[i][s]
                        self.stream_f_g[ip][s] = self.collide_f_g[i][s]
                    else:
                        self.stream_f_l[i][self.REVERSED_E[s]] = self.collide_f_l[i][s]
                        self.stream_f_g[i][self.REVERSED_E[s]] = self.collide_f_g[i][s]

    def export_VTK(self, n):

        grid_x = np.linspace(0, lx, lx)
        grid_y = np.linspace(0, ly, ly)
        grid_z = np.linspace(0, lz, lz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        gridToVTK(
            "./G" + str(round(G_ls, 4)) + "_" + str(n),
            grid_x,
            grid_y,
            grid_z,
            pointData={
                "Solid": np.ascontiguousarray(
                    self.IS_SOLID.to_numpy()[0:lx, 0:ly, 0:lz]
                ),
                "rho_l": np.ascontiguousarray(self.rho_l.to_numpy()[0:lx, 0:ly, 0:lz]),
                "rho_g": np.ascontiguousarray(self.rho_g.to_numpy()[0:lx, 0:ly, 0:lz]),
                "pressure": np.ascontiguousarray(
                    self.pressure.to_numpy()[0:lx, 0:ly, 0:lz]
                ),
                "velocity_l": (
                    np.ascontiguousarray(self.v_l.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.v_l.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.v_l.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "velocity_g": (
                    np.ascontiguousarray(self.v_g.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.v_g.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.v_g.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "force_l": (
                    np.ascontiguousarray(self.force_l.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.force_l.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.force_l.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
                "force_g": (
                    np.ascontiguousarray(self.force_g.to_numpy()[0:lx, 0:ly, 0:lz, 0]),
                    np.ascontiguousarray(self.force_g.to_numpy()[0:lx, 0:ly, 0:lz, 1]),
                    np.ascontiguousarray(self.force_g.to_numpy()[0:lx, 0:ly, 0:lz, 2]),
                ),
            },
        )

    def run(self, max_step=20000):

        self.init_field()
        while self.step[None] < max_step:

            self.update_psi()
            self.update_force()
            self.update_velocity()
            self.collision()
            self.streaming()
            self.post_collsion()

            if (self.step[None]) % vtk_dstep == 0:
                self.export_VTK(self.step[None] // vtk_dstep)
                print(
                    "Export No.{} vtk at step {}".format(
                        self.step[None] // vtk_dstep, self.step[None]
                    )
                )

            self.step[None] += 1


# In[2]:


MRT = lbm_single_phase(sparse_mem=False)
MRT.init_field()
print("{} fluid nodes have been activated!".format(MRT.activity_checking()))
print("{} fluid nodes in the computational domain!".format(lx * ly * lz - solid_count))


# In[3]:


begin = time.time()
MRT.run(100)
print("Time elapses for 100 steps using non-sparse memory is ", time.time() - begin)


# In[4]:


MRT = lbm_single_phase(sparse_mem=True)
MRT.init_field()
print("{} fluid nodes have been activated!".format(MRT.activity_checking()))
print("{} fluid nodes in the computational domain!".format(lx * ly * lz - solid_count))


# In[5]:


begin = time.time()
MRT.run(100)
print("Time elapses for 100 steps using sparse memory is ", time.time() - begin)

