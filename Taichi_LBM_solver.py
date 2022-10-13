import taichi as ti
import numpy as np
import math
from sympy import inverse_mellin_transform
from pyevtk.hl import gridToVTK


ti.init(arch=ti.cpu,dynamic_index=True)

# LBM parameters
Q = 19
half = (Q - 1) // 2

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

# reversed_e_xyz_np stores the index of the opposite component to every component in e_xyz_np
# For example, for [1,0,0], the opposite component is [-1,0,0] which has the index of 2 in e_xyz
reversed_e_np = np.array([e_xyz_list.index([-a for a in e]) for e in e_xyz_list])
print(reversed_e_np)

# Predefined compound types
i32_vec3d = ti.types.vector(3, ti.i32)
f32_vec3d = ti.types.vector(3, ti.f32)

# MRT operator
niu = 0.1
tau_f=3.0*niu+0.5
s_v=1.0/tau_f
s_other=8.0*(2.0-s_v)/(8.0-s_v)
S_dig_np = np.array([0,s_v,s_v,0,s_other,0,s_other,0,s_other, s_v, s_v,s_v,s_v,s_v,s_v,s_v,s_other,s_other,s_other])

# M_np = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
# [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
# [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
# [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
# [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
# [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
# [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
# [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
# [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
# [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
# [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
# [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
# [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
# [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
# [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
# [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
# [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]])

M_np = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[-30,-11,-11,-11,-11,-11,-11,8,8,8,8,8,8,8,8,8,8,8,8],
[12,-4,-4,-4,-4,-4,-4,1,1,1,1,1,1,1,1,1,1,1,1],
[0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0],
[0,-4,4,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0],
[0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1],
[0,0,0,-4,4,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1],
[0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1],
[0,0,0,0,0,-4,4,0,0,1,-1,1,-1,0,0,-1,1,-1,1],
[0,2,2,-1,-1,-1,-1,1,1,1,1,-2,-2,1,1,1,1,-2,-2],
[0,-4,-4,2,2,2,2,1,1,1,1,-2,-2,1,1,1,1,-2,-2],
[0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,1,1,-1,-1,0,0],
[0,0,0,-2,-2,2,2,1,1,-1,-1,0,0,1,1,-1,-1,0,0],
[0,0,0,0,0,0,0,1,1,0,0,0,0,-1,-1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,-1,-1],
[0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,-1,-1,0,0],
[0,0,0,0,0,0,0,1,-1,-1,1,0,0,1,-1,-1,1,0,0],
[0,0,0,0,0,0,0,-1,1,0,0,1,-1,1,-1,0,0,1,-1],
[0,0,0,0,0,0,0,0,0,1,-1,-1,1,0,0,-1,1,1,-1]])

inv_M_np = np.linalg.inv(M_np)

# Input paramters
lx = ly = lz = 120
x = np.linspace(0, lx, lx)
y = np.linspace(0, ly, ly)
z = np.linspace(0, lz, lz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

IniPerturbRate = 1
rho0 = 0.25
carn_star = True
T_Tc = 0.7
G = -1.0
inject_type = 2  # 0: fluid nodes, 1: gas nodes, 2: liquid nodes
rho_inject_period = 2000
rho_increment = 0.002
rhol_spinodal = 0.2725
rhog_spinodal = 0.0484
rhos = 0.35

tau = 1.0  # specify the relaxaton time (only for BGK operator)
inv_tau = 1/tau

# BRK = False
A = tau
MRT = True #if false, we use BGK operator instead
if MRT: # Refer to Table 6.1 in Kruger's book
    A = 0.5

# Writing input model (here we create 8 20-lu-diameter spheres which are uniformly stacked)
grain_diameter = 30
grain_number = math.floor(lx / grain_diameter)
with open(
    "./lx" + str(lx) + "_" + str(int(grain_diameter)),
    "w",
) as f:
    for i in range(grain_number):
        x = grain_diameter / 2 + i * grain_diameter
        for j in range(grain_number):
            y = grain_diameter / 2 + j * grain_diameter
            for k in range(grain_number):
                z = grain_diameter / 2 + k * grain_diameter
                f.write(str(x) + "\n")
                f.write(str(y) + "\n")
                f.write(str(z) + "\n")
                f.write(str(grain_diameter / 2 * 1.01) + "\n")


solid_np = np.zeros((lx, ly, lz), dtype=np.int8)
solid_count = 0

def place_sphere(x, y, z, R):

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


read_positions("./lx" + str(lx) + "_" + str(int(grain_diameter)))
print(
    "The computational domain has {} grains with {} lu in diameter.".format(
        solid_count, grain_diameter
    ),
)


get_ipython().run_line_magic('load_ext', 'nb_black')
from zipimport import zipimporter


@ti.data_oriented
class lbm_single_phase:
    def __init__(self):

        self.step = 0
        self.inject_type = inject_type

        self.nb_solid_nodes = ti.field(ti.i32,shape=())
        self.nb_fluid_nodes = ti.field(ti.i32,shape=())
        self.saturation = ti.field(ti.f32,shape=())
        self.suction = ti.field(ti.f32,shape=())
        
        # self.nb_solid_nodes[None] = 0
        # self.nb_fluid_nodes[None] = 0
        # self.saturation[None] = 0.
        # self.suction[None] = 0.

        self.pressure = ti.field(ti.f32)
        self.force = ti.Vector.field(3, ti.f32)
        self.v = ti.Vector.field(3, ti.f32)

        self.collide_f = ti.Vector.field(Q, ti.f32)
        self.stream_f = ti.Vector.field(Q, ti.f32)

        self.is_solid = ti.field(ti.i8,shape=(lx, ly, lz))
        self.rho = ti.field(ti.f32,shape=(lx, ly, lz))
    
        n_mem_partition = 3  # Generate blocks of 3X3x3

        cell1 = ti.root.pointer(ti.ijk, (lx//n_mem_partition+1,ly//n_mem_partition+1,lz//n_mem_partition+1))

        # cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.v)
        # cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.pressure)
        # cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.force)
        # cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.collide_f)
        # cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.stream_f)

        cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.v)
        cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.pressure)
        cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.force)
        cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.collide_f)
        cell1.dense(ti.ijk, (n_mem_partition,n_mem_partition,n_mem_partition)).place(self.stream_f)

        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(19,19,ti.f32, shape=())
        self.S_dig = ti.Vector.field(19,ti.f32,shape=())
        self.e_xyz = ti.Vector.field(3, dtype=ti.i32, shape=(Q))
        self.e_xyz.from_numpy(np.array(e_xyz_list))
        self.reversed_e_index = ti.field(dtype=ti.i32, shape=(Q))
        self.reversed_e_index.from_numpy(reversed_e_np)
        
        self.M[None] = ti.Matrix(M_np)
        self.inv_M[None] = ti.Matrix(inv_M_np)
        self.S_dig.from_numpy(S_dig_np)
        self.is_solid.from_numpy(solid_np)

        ti.static(self.inv_M)
        ti.static(self.is_solid)
        ti.static(self.S_dig)
        ti.static(self.M)
        ti.static(self.e_xyz)
        ti.static(self.reversed_e_index)

    @ti.func
    def Press(self, rho_value) -> ti.f32:
        if ti.static(carn_star):
            a = 1.0
            b = 4.0
            R = 1.0
            Tc = 0.0943
            T = T_Tc * Tc
            eta = b * rho_value / 4.0
            eta2 = eta * eta
            eta3 = eta2 * eta
            rho2 = rho_value * rho_value
            one_minus_eta = 1.0 - eta
            one_minus_eta3 = one_minus_eta * one_minus_eta * one_minus_eta
            return (
                rho_value * R * T * (1 + eta + eta2 - eta3) / one_minus_eta3 - a * rho2
            )

        else:
            cs2 = 1.0 / 3.0
            psi = 1.0 - ti.exp(-rho_value)
            psi2 = psi * psi
            return cs2 * rho_value + cs2 * G / 2 * psi2

    @ti.func
    def Psi(self, rho_value) -> ti.f32:
        if ti.static(carn_star):
            cs2 = 1.0 / 3.0
            p = self.Press(rho_value)
            return ti.sqrt(2.0 * (p - cs2 * rho_value) / (cs2 * G))
        else:
            return 1.0 - ti.exp(-rho_value)

    @ti.func
    def force_vec(self,local_pos) -> f32_vec3d:
        force_vec = ti.Vector([0., 0.,0.])
        # local_pos = ti.Vector([x,y,z])
        local_psi = self.Psi(self.rho[local_pos])
        for i in ti.static(range(3)):
            for s in ti.static(range(1,Q)):    
                neighbor_pos = self.periodic_index(local_pos+self.e_xyz[s])
                neighbor_psi = self.Psi(self.rho[neighbor_pos])
                force_vec[i] += (w[s] * neighbor_psi * self.e_xyz[s][i])
        force_vec *=(-G* local_psi) 
        return force_vec

    @ti.func
    def velocity_vec(self,local_pos) -> f32_vec3d:

        velocity_vec = ti.Vector([0., 0.,0.])
        for i in ti.static(range(3)):
            for s in ti.static(range(Q)):
                velocity_vec[i] += (self.collide_f[local_pos][s]* self.e_xyz[s][i])
           
            velocity_vec[i] += A*self.force[local_pos][i]
            
            velocity_vec[i]/=self.rho[local_pos]

        return velocity_vec

    @ti.func
    def meq_vec(self,rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        
        # Taichi-LBM3D: A Single-Phase and Multiphase Lattice Boltzmann Solver on Cross-Platform Multicore CPU/GPUs
        # out[0] = rho_local
        # out[3] = u[0]
        # out[5] = u[1]
        # out[7] = u[2]
        # out[1] = u.dot(u)    
        # out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z
        # out[11] = u.y*u.y-u.z*u.z
        # out[13] = u.x*u.y
        # out[14] = u.y*u.z                            
        # out[15] = u.x*u.z

        # The D3Q19 Gram-Schmidt equilibrium moments
        out[0] = 1
        out[1] = -11+19*u.dot(u)
        out[2] = 3-11/2*u.dot(u)
        out[3] = u.x
        out[4] = -2/3*u.x
        out[5] = u.y
        out[6] = -2/3*u.y
        out[7] = u.z
        out[8] = -2/3*u.z
        out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z
        out[10] = -out[9]/2
        out[11] = u.y*u.y-u.z*u.z
        out[12] = -out[11]/2
        out[13] = u.x*u.y
        out[14] = u.y*u.z
        out[15] = u.x*u.z
        return out*rho_local

    @ti.kernel
    def init_field(self):

        self.nb_fluid_nodes[None] = 0

        for x,y,z in self.is_solid: 
            self.rho[x,y,z] = rhos      

            if self.is_solid[x,y,z] == 0:
                self.rho[x,y,z] = rho0 * (1.0 + IniPerturbRate * (ti.random(ti.f32) - 0.5))
                self.nb_fluid_nodes[None] +=1

                for q in ti.static(range(Q)):
                    self.collide_f[x,y,z][q] = t[q] * self.rho[x,y,z]
                    self.stream_f[x,y,z][q] = t[q] * self.rho[x,y,z]
            
            else:
                self.rho[x,y,z] = rhos

    # check if sparse storage works!
    @ti.kernel
    def activity_checking(self):
        self.nb_fluid_nodes[None] = 0
        self.nb_solid_nodes[None] = 0
        for x,y,z in self.collide_f:
            if x<lx and y<ly and z<lz:
                self.nb_fluid_nodes[None] +=1
                if self.is_solid[x,y,z] == 1:
                    self.nb_solid_nodes[None] +=1
        self.nb_fluid_nodes[None] -= self.nb_solid_nodes[None]
    
    @ti.kernel
    def collision(self):
        for I in ti.grouped(self.collide_f):
            if (I.x < lx and I.y<ly and I.z<lz and self.is_solid[I] == 0):
                self.force[I] = self.force_vec(I)
                self.v[I] = self.velocity_vec(I)

                if ti.static(MRT):
                    """MRT operator"""
                    Mxf = self.M[None]@self.collide_f[I] 
                    m_eq = self.meq_vec(self.rho[I],self.v[I])
                    SxMf_minus_meq = self.S_dig[None]*(m_eq-Mxf)
                    self.collide_f[I] += self.inv_M[None]@SxMf_minus_meq
                    
                    # Method 1：m_eq = M@f_eq
                    # f_eq = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                    # u_squ = self.v[I].dot(self.v[I])
                    # for s in ti.static(range(Q)):
                    #     eu = self.e_xyz[s].dot(self.v[I])
                    #     f_eq[s] = t[s] * self.rho[I] *(1.0 + 3.0 * eu \
                    #         + 4.5 * eu * eu - 1.5 * u_squ) -self.collide_f[I][s]
                    # Mxf_eq = self.M[None]@f_eq # M*f = M@f_eq
                    
                    # Method 2: 
                    # meq = self.meq_vec(self.rho[I],self.v[I]) # m_eq = M@f_eq
                    # t[s] * self.rho[I] *(1.0 + 3.0 * eu \
                    #         + 4.5 * eu * eu - 1.5 * u_squ) -self.collide_f[I][s]
                    # m_temp = -self.S_dig[None]*(m_temp-meq)
                    # for s in ti.static(range(Q)):
                    #     f_guo=0.0
                    #     for l in ti.static(range(Q)):
                    #         f_guo += w[l]*((self.e_xyz[l]-self.v[I]).dot(self.force[I])+\
                    #             (self.e_xyz[l].dot(self.v[I])*(self.e_xyz[l].dot(self.force[I]))))*self.M[None][s,l]
                    #     m_temp[s] += (1-0.5*self.S_dig[None][s])*f_guo
                    # self.collide_f[I] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                    # self.collide_f[I] += self.inv_M[None]@m_temp
                
                else:
                    """BGK operator"""
                    u_squ = self.v[I].dot(self.v[I])
                    for s in ti.static(range(Q)):
                        eu = self.e_xyz[s].dot(self.v[I])
                        self.collide_f[I][s] +=(t[s] * self.rho[I] *(1.0 + 3.0 * eu                             + 4.5 * eu * eu - 1.5 * u_squ) -self.collide_f[I][s]) *inv_tau
              


    @ti.kernel
    def post_collsion(self):
        """Calculate force and velocity"""
        for I in ti.grouped(self.collide_f):
            if (I.x < lx and I.y<ly and I.z<lz and self.is_solid[I] == 0):
                    self.collide_f[I] = self.stream_f[I]
                    self.rho[I] = self.collide_f[I].sum()

                    if self.step % rho_inject_period == 0 and self.step:
                            if self.inject_type == 0:
                                self.rho[I] += rho_increment
                            elif self.inject_type == 1:
                                if self.rho[I] < rhol_spinodal:
                                    self.rho[I] += rho_increment
                            else:
                                if self.rho[I] >= rhol_spinodal:
                                    self.rho[I] += rho_increment

    @ti.func
    def periodic_index(self,i):
        iout = i
        if i[0]<0:     iout[0] = lx-1
        if i[0]>lx-1:  iout[0] = 0
        if i[1]<0:     iout[1] = ly-1
        if i[1]>ly-1:  iout[1] = 0
        if i[2]<0:     iout[2] = lz-1
        if i[2]>lz-1:  iout[2] = 0

        return iout

    @ti.kernel
    def streaming(self):
        for I in ti.grouped(self.collide_f):
            if (I.x < lx and I.y<ly and I.z<lz and self.is_solid[I] == 0):
                for s in ti.static(range(19)):
                    neighbor_pos = self.periodic_index(I+self.e_xyz[s])
                    if (self.is_solid[neighbor_pos]==0):
                        # Push scheme:
                        self.stream_f[neighbor_pos][s] = self.collide_f[I][s]
                        # Pull scheme:
                        # self.stream_f[xyz,s] = self.collide_f[neighbor_xyz,s]
                    else:
                        self.stream_f[I][self.reversed_e_index[s]] = self.collide_f[I][s]
        
    def export_VTK(self, n):
        
        grid_x = np.linspace(0, lx, lx)
        grid_y = np.linspace(0, ly, ly)
        grid_z = np.linspace(0, lz, lz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        gridToVTK(
                "./LB_SingelPhase_"+str(n),
                grid_x,
                grid_y,
                grid_z,
                pointData={ "Solid": np.ascontiguousarray(self.is_solid.to_numpy()[0:lx,0:ly,0:lz]),
                            "rho": np.ascontiguousarray(self.rho.to_numpy()[0:lx,0:ly,0:lz]),
                            "pressure": np.ascontiguousarray(self.pressure.to_numpy()[0:lx,0:ly,0:lz]),
                            "velocity": (   np.ascontiguousarray(self.v.to_numpy()[0:lx,0:ly,0:lz,0]), 
                                            np.ascontiguousarray(self.v.to_numpy()[0:lx,0:ly,0:lz,1]), 
                                            np.ascontiguousarray(self.v.to_numpy()[0:lx,0:ly,0:lz,2]))
                            }
            )   

    @ti.kernel
    def calc_stat(self):
        nb_liq_nodes = 0
        nb_vap_nodes = 0
        liq_pressure = 0.
        vap_pressure = 0.
        average_liq_pressure = 0.
        average_vap_pressure = 0.
        for I in ti.grouped(self.collide_f):
            if (I.x < lx and I.y<ly and I.z<lz and self.is_solid[I] == 0):
                self.pressure[I] = self.Press(self.rho[I])
                if self.rho[I]>=rhol_spinodal:
                    nb_liq_nodes += 1
                    liq_pressure += self.pressure[I]
                elif self.rho[I]<=rhog_spinodal:
                    nb_vap_nodes +=1
                    vap_pressure += self.pressure[I]
                
        self.saturation[None] =nb_liq_nodes/self.nb_fluid_nodes[None]
        
        if nb_liq_nodes:
            average_liq_pressure = liq_pressure/nb_liq_nodes
        if nb_vap_nodes:
            average_vap_pressure = vap_pressure/nb_vap_nodes
        
        self.suction[None] = average_vap_pressure - average_liq_pressure
    
    def run(self):
        self.init_field()
        
        print(self.nb_fluid_nodes[None])
        while self.step < 5000:    
            self.collision()
            self.streaming()
            self.post_collsion()
        
            if self.step%100 == 0:
                self.calc_stat()
                print("Saturation is {} and suction is {} at step {}".format(self.saturation[None],self.suction[None],self.step))
                self.export_VTK(self.step//100)
                print("Export No.{} vtk at step {}".format(self.step//20,self.step))
           
            self.step+=1 



example = lbm_single_phase()
example.run()

