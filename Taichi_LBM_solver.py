import taichi as ti
import numpy as np
import math
from sympy import inverse_mellin_transform
from pyevtk.hl import gridToVTK

ti.init(arch=ti.cpu, dynamic_index=True, cpu_max_num_threads=36)

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
e_xyz_np = np.array(
    [
        [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
    ]
)
e_xyz = ti.Vector.field(Q, dtype=ti.f32, shape=(3))
e_xyz.from_numpy(e_xyz_np)

# Predefined compound types
i32_vec3d = ti.types.vector(3, ti.i32)

# Input paramters
lx = ly = lz = 100
IniPerturbRate = 1
rho0 = 0.2
carn_star = True
T_Tc = 0.7
G = -1.0
inject_type = 0  # 0: fluid nodes, 1: gas nodes, 2: liquid nodes
rho_inject_period = 1000
rho_increment = 0.005
rhol_spinodal = 0.2725
rhog_spinodal = 0.0484
rhos = 0.35
tau = 1.0  # specify the relaxaton time (only for BGK operator)


# Writing input model (here we create 8 20-lu-diameter spheres which are uniformly stacked)
grain_diameter = 25
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
is_solid_np = np.zeros((lx, ly, lz), dtype=np.int32)
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
                    is_solid_np[near_px, near_py, near_pz] = 1
                    
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
is_solid = ti.field(ti.i32, shape=(lx, ly, lz))
is_solid.from_numpy(is_solid_np)


@ti.data_oriented
class lbm_single_phase:
    def __init__(self):

        self.step = 0
        self.inject_type = inject_type

        self.nb_solid_nodes = ti.field(ti.i32,shape=())
        self.nb_fluid_nodes = ti.field(ti.i32,shape=())
        self.saturation = ti.field(ti.f32,shape=())
        self.suction = ti.field(ti.f32,shape=())

        self.collide_f = ti.Vector.field(19, ti.f32)
        self.stream_f = ti.Vector.field(19, ti.f32)
        self.rho = ti.field(ti.f32)
        self.pressure = ti.field(ti.f32)
        self.psi = ti.field(ti.f32)
        self.force = ti.Vector.field(3, ti.f32)
        self.v = ti.Vector.field(3, ti.f32)
        force.fill(0.)
 
        n_mem_partition = 2  # Generate blocks of 2X2x2

        block = ti.root.pointer(
            ti.ijk,
            (
                lx // n_mem_partition,
                ly // n_mem_partition,
                lz // n_mem_partition,
            ),
        )
        self.cell = block.bitmasked(
            ti.ijk, (n_mem_partition, n_mem_partition, n_mem_partition)
        )# dense or bitmasked
        self.cell.place(
            self.rho,
            self.pressure,
            self.collide_f,
            self.stream_f,
            self.psi,
            self.force,
            self.v, 
        )

        self.S_dig = ti.Vector.field(19,ti.f32,shape=())
        
        self.niu = 0.16667
        self.tau_f=3.0*self.niu+0.5
        self.s_v=1.0/self.tau_f
        self.s_other=8.0*(2.0-self.s_v)/(8.0-self.s_v)

        self.S_dig[None] = ti.Vector([0,self.s_v,self.s_v,0,self.s_other,0,            self.s_other,0,self.s_other, self.s_v, self.s_v,self.s_v,self.s_v,                self.s_v,self.s_v,self.s_v,self.s_other,self.s_other,self.s_other])
        self.M = ti.Matrix.field(19, 19, ti.f32, shape=())
        self.inv_M = ti.Matrix.field(19,19,ti.f32, shape=())
        # self.M = ti.field(ti.f32, shape=(19,19))
        # self.inv_M = ti.field(ti.f32, shape=(19,19))        
        M_np = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
        [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
        [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
        [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]])
        inv_M_np = np.linalg.inv(M_np)
        self.M[None] = ti.Matrix([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
        [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
        [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
        [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
        [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
        [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
        [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
        [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]])
        self.inv_M[None] = ti.Matrix(inv_M_np)
        
        ti.static(self.inv_M)
        ti.static(self.M)
        ti.static(self.S_dig)

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

    @ti.kernel
    def init_field(self):
        for x,y,z in ti.ndrange(lx, ly, lz):
            if is_solid[x,y,z]:
                for q in ti.static(range(Q)):
                    next_x, next_y, next_z = self.neighbor_node(x,y,z,q)
                    if is_solid[next_x, next_y, next_z] == 0:
                        self.rho[x,y,z] = rhos
                        self.psi[x,y,z] = self.Psi(rhos)
                        
            else is_solid[x,y,z] == 0:
                self.rho[x,y,z] = rho0 * (
                    1.0 + IniPerturbRate * (ti.random(dtype=ti.f32) - 0.5)
                )
                for q in ti.static(range(Q)):
                    self.collide_f[x,y,z][q] = t[q] * self.rho[x,y,z]
                    self.stream_f[x,y,z][q] = t[q] * self.rho[x,y,z]

    # check if sparse storage works!
    @ti.kernel
    def activity_checking(self):
        nb_active_nodes = 0
        nb_solid_nodes = 0
        for x,y,z in self.rho:
            if x < lx and y < ly and z< lz:
                nb_active_nodes += 1
                if is_solid[x,y,z]:
                    self.nb_solid_nodes[None] +=1
        self.nb_fluid_nodes[None] = nb_active_nodes-self.nb_solid_nodes[None]

    @ti.kernel
    def collision(self):
        """Update fluid density"""
        for x,y,z in self.rho:
            if is_solid[x,y,z] == 0:
                self.rho[x,y,z] = self.stream_f[x,y,z].sum()

                if self.step % rho_inject_period == 0 and self.step:
                    if self.inject_type == 0:
                        self.rho[x,y,z] += rho_increment
                    elif self.inject_type == 1:
                        if self.rho[x,y,z] < rhol_spinodal:
                            self.rho[x,y,z] += rho_increment
                    else:
                        if self.rho[x,y,z] >= rhol_spinodal:
                            self.rho[x,y,z] += rho_increment

                self.pressure[x,y,z] = self.Press(self.rho[x,y,z])
                self.psi[x,y,z] = self.Psi(self.rho[x,y,z])

    @ti.func
    def meq_vec(self,rho_local,u):
        out = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        out[0] = rho_local
        out[1] = u.dot(u) 
        out[3] = u[0]    
        out[5] = u[1]    
        out[7] = u[2]
        out[9] = 2*u.x*u.x-u.y*u.y-u.z*u.z        
        out[11] = u.y*u.y-u.z*u.z
        out[13] = u.x*u.y
        out[14] = u.y*u.z
        out[15] = u.x*u.z
        return out

    @ti.kernel
    def post_collsion(self):
        u_x = u_y = u_z = 0.0

        """Calculate force and velocity"""
        for x, y, z in self.force:
            if is_solid[x, y, z] == 0 and x<lx and y<ly and z<lz:
                xp = (x > 0) if (x - 1) else (lx - 1)
                xn = (x < lx - 1) if (x + 1) else (0)
                yp = (y > 0) if (y - 1) else (ly - 1)
                yn = (y < ly - 1) if (y + 1) else (0)
                zp = (z > 0) if (z - 1) else (lz - 1)
                zn = (z < lz - 1) if (z + 1) else (0)

                for i in ti.static(range(3)):
                    self.force[x, y, z][i] = (
                        -G* self.psi[x, y, z]* (w[1] * self.psi[xn, y, z] * e_xyz[i][1]
                            + w[2] * self.psi[xp, y, z] * e_xyz[i][2]
                            + w[3] * self.psi[x, yn, z] * e_xyz[i][3]
                            + w[4] * self.psi[x, yp, z] * e_xyz[i][4]
                            + w[5] * self.psi[x, y, zn] * e_xyz[i][5]
                            + w[6] * self.psi[x, y, zp] * e_xyz[i][6]
                            + w[7] * self.psi[xn, yn, z] * e_xyz[i][7]
                            + w[8] * self.psi[xp, yp, z] * e_xyz[i][8]
                            + w[9] * self.psi[xn, yp, z] * e_xyz[i][9]
                            + w[10] * self.psi[xp, yn, z] * e_xyz[i][10]
                            + w[11] * self.psi[xn, y, zn] * e_xyz[i][11]
                            + w[12] * self.psi[xp, y, zp] * e_xyz[i][12]
                            + w[13] * self.psi[xn, y, zp] * e_xyz[i][13]
                            + w[14] * self.psi[xp, y, zn] * e_xyz[i][14]
                            + w[15] * self.psi[x, yn, zn] * e_xyz[i][15]
                            + w[16] * self.psi[x, yp, zp] * e_xyz[i][16]
                            + w[17] * self.psi[x, yn, zp] * e_xyz[i][17]
                            + w[18] * self.psi[x, yp, zn] * e_xyz[i][18]
                        ))

                        
                    self.v[x, y, z][i] = (self.stream_f[x, y, z] * e_xyz[i]).sum()+ self.force[x, y, z][i]/2 
                    
                    # self.v[x, y, z][i] = (self.stream_f[x, y, z] * e_xyz[i]).sum()+ self.force[x,y,z][i] * tau #if it's BGK operator!
                    # inv_rho = 1.0 / self.rho[x, y, z]

                    # self.v[x, y, z][i] *= inv_rho

                # BGK operator
                # u_squ = self.v[x, y, z][0]*self.v[x, y, z][0] +\
                #     self.v[x, y, z][1]*self.v[x, y, z][1]+self.v[x, y, z][2]*self.v[x, y, z][2]
                
                # for i in ti.static(range(Q)):
                #     eu = e_xyz[0][i] * self.v[x, y, z][0] + e_xyz[1][i] * self.v[x, y, z][1] + e_xyz[2][i] * self.v[x, y, z][2] 
                #     self.stream_f[x,y,z][i] += (t[i]*self.rho[x,y,z]*(1.0 + 3.0 * eu + 4.5 * eu * eu \
                #         - 1.5 * u_squ)-self.stream_f[x,y,z][i])/tau  

                # MRT operator   
                m_temp = self.M[None]@self.stream_f[x, y, z]
                meq = self.meq_vec(self.rho[x, y, z],self.v[x, y, z])
                m_temp -= self.S_dig[None]*(m_temp-meq)
                
                self.stream_f[x, y, z] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                self.stream_f[x, y, z] += self.inv_M[None]@m_temp


    @ti.kernel
    def swap_f(self):
        for I in ti.grouped(self.stream_f):
            temp = self.stream_f[I]
            self.stream_f[I] = self.collide_f[I]
            self.collide_f[I] = temp
    
    @ti.func
    def neighbor_node(self,x:ti.i32,y:ti.i32,z:ti.i32,i:ti.i32):
        next_x = x - e_xyz[0][i]
        if x == 0 and e_xyz[0][i] == 1:
            next_x = lx - 1
        if x == lx - 1 and e_xyz[0][i] == -1:
            next_x = 0

        next_y = y - e_xyz[1][i]
        if y == 0 and e_xyz[1][i] == 1:
            next_y = ly - 1
        if y == ly - 1 and e_xyz[1][i] == -1:
            next_y = 0   

        next_z = z - e_xyz[2][i]
        if z == 0 and e_xyz[2][i] == 1:
            next_z = lz - 1
        if z == lz - 1 and e_xyz[2][i] == -1:
            next_z = 0 
            
        return int(next_x),int(next_y),int(next_z)

    @ti.kernel
    def bounce_back(self):
        for I in ti.grouped(self.collide_f):
            self.stream_f[I][0] = self.collide_f[I][0]
            if is_solid[I.x,I.y,I.z]==0:
                for i in ti.static(range(1,Q)):
                    next_x, next_y, next_z = self.neighbor_node(I.x,I.y,I.z,i)
                    if (is_solid[next_x,next_y,next_z]):
                        switched_i = (i - half) if i > half else (i + half)
                        self.stream_f[I][i] = self.collide_f[I][switched_i]

    @ti.kernel
    def streaming(self):
        for I in ti.grouped(self.collide_f):
            if is_solid[I.x,I.y,I.z]==0 and I.x<lx and I.y<ly and I.z<lz:
                for i in ti.static(range(1,Q)):
                    next_x, next_y, next_z = self.neighbor_node(I.x,I.y,I.z,i)
                    if is_solid[next_x, next_y, next_z] ==0:
                        self.stream_f[I][i] = self.collide_f[next_x, next_y, next_z][i]
    
    def export_VTK(self, n):
        x = np.linspace(0, lx, lx)
        y = np.linspace(0, ly, ly)
        z = np.linspace(0, lz, lz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        gridToVTK(
                "./LB_SingelPhase_"+str(n),
                x,
                y,
                z,
                pointData={ "Solid": np.ascontiguousarray(is_solid_np),
                            "rho": np.ascontiguousarray(self.rho.to_numpy()),
                            "pressure": np.ascontiguousarray(self.pressure.to_numpy()),
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
        for I in ti.grouped(self.rho):
            if is_solid[I.x,I.y,I.z]==0:
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
        self.activity_checking()
        self.export_VTK(0)
        
        while self.step < 2000:
            self.streaming()
            self.bounce_back()
            self.collision()
            self.post_collsion()
            self.swap_f()
            self.step+=1
            if self.step%20 == 0:
                self.calc_stat()
                print("Saturation is {} and suction is {} at step {}".format(self.saturation[None],self.suction[None],self.step))
                self.export_VTK(self.step//20+1)
                print("Export No.{} vtk at step {}".format(self.step//20,self.step))


example = lbm_single_phase()

example.step=0
example.run()

