"""
---Bindesh Tripathi
June 19, 2023.
"""


import numpy as np
import h5py
from mpi4py import MPI
CW = MPI.COMM_WORLD
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import time
import shutil, os



import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)


import numpy as np
import dedalus.public as de
import matplotlib.pyplot as plt

from dedalus.core.basis import Fourier
from dedalus.core.field import Operand
from dedalus.core.operators import Separable, FutureField
from dedalus.tools.array import reshape_vector


class FractionalLaplacian(Separable, FutureField):
    """
    Fourier fractional Laplacian operator: (-Δ)**s

    Parameters
    ----------
    arg : field object
        Field argument
    s : float
        Laplacian power

    Notes
    -----
    The fractional Laplacian is defined as (-Δ)**s, with a corresponding
    Fourier symbol |k|**(2s).  The standard Laplacian is recovered, up to an
    overall negative sign, with s=1.

    See https://www.ma.utexas.edu/mediawiki/index.php/Fractional_Laplacian

    """

    def __new__(cls, arg0, *args, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Check all bases are Fourier
        for basis in arg0.domain.bases:
            if not isinstance(basis, Fourier):
                raise NotImplementedError("Operator only implemented for full-Fourier domains. ")
        # Check for scalars
        if arg0.domain.dim == 0:
            return 0
        else:
            return object.__new__(cls)

    def __init__(self, arg, s, **kw):
        arg = Operand.cast(arg)
        super().__init__(arg, **kw)
        self.kw = {'s': s}
        self.s = s
        self.name = 'Lap[%s]' % self.s
        self.axis = None
        # Build operator symbol array
        slices = self.domain.dist.coeff_layout.slices(self.domain.dealias)
        local_wavenumbers = [self.domain.elements(axis) for axis in range(self.domain.dim)]
        local_k2 = np.sum([ki**2 for ki in local_wavenumbers], axis=0)
        local_k2_mod = local_k2.copy()
        local_k2_mod[local_k2 == 0] = 1
        self.local_symbols = local_k2_mod ** s
        self.local_symbols[local_k2 == 0] = 0

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def check_conditions(self):
        arg0, = self.args
        # Must be in coeff layout
        is_coeff = not np.any(arg0.layout.grid_space)
        return is_coeff

    def operator_form(self, index):
        # Get local index, special casing zero mode for NCC preconstruction pass
        if any(index):
            local_index = index - self.domain.dist.coeff_layout.start(scales=None)
        else:
            local_index = index
        return self.local_symbols[tuple(local_index)]

    def operate(self, out):
        arg0, = self.args
        # Require coeff layout
        arg0.require_coeff_space()
        out.layout = arg0.layout
        # Apply symbol array to coefficients
        np.multiply(arg0.data, self.local_symbols, out=out.data)


# Add operator to namespace
de.operators.parseables['FractionalLaplacian'] = FractionalLaplacian

#==============================================================================================================
#==============================================================================================================
#CHOOSE YOUR PARAMETERS
#==============================================================================================================
#==============================================================================================================
hydro_or_mhd         = 'mhd' #'hydro', 'mhd'
MA                   = 60.0      #40 #2.5 #Alfven Mach number, defined as U_0/v_A
Reynolds             = 15 
MagReynolds          = 15
D_Krook              = 2.0       #0 #0.1  
Schmidt              = 1.0
Lx                   = 10.0*np.pi
Ly                   = 10.0*np.pi
Lz                   = 20.0*np.pi
nx                   = 128  #512 #96 #256*2 #1024  #192
ny                   = 128  #512 #96 #256*2 #1024  #192
nz                   = 1024  #512 #96 #1024  #256 #96
mesh                 = [32, 64] #no. of processros along x and y directions. product of this should match with number of total processir asked.   
sim_time             = 1001     #301 #82.00 #148.00 #65.55 #45.01 #make this 1 more than a perfect even number so that  sim_time = integer*checkpoint_time +1.
checkpoint_time      = 20
wall_time            = 86400/24*86 #The last one hour out of 48 hours would be devoted to merge files at the end of solver calculations.
max_steps            = np.inf 
forced               = True   #False #True
force_nature         = 'force2'
passive_scalar       = False    #False #True
initcond             = 'noise' #'coherent'  #'noise'
checkpoint_read      = True
run_number           = 'run539'  #run1 #used to store files for separate runs in separate directory
month_of_run         = 'oct2020'
script_version_number= run_number # change script title as well with the change in version.
submit_version_number= script_version_number     # change submit.cmd file as well with the change in version. 
                                                 # This feature is added to allow multiple queue jobs in sbatch Stampede2

amp                  = 1e-8
sigma                = 2
a                    = 1
z1                   = 1/4*Lz
z2                   = 3/4*Lz

# restart_after_core_crash = False




# ######ICs are imported here.
# if restart_after_core_crash == False:
#     nx_IC                = 16 #gridresolution in x-direction in IC
#     ny_IC                = 16 #gridresolution in y-direction in IC
#     nz_IC                = 512 #gridresolution in z-direction in IC


# #Finding where the IC data is located.
# if hydro_or_mhd == 'mhd':
#     if restart_after_core_crash == False:
#         IC_filename_loc      = "/anvil/projects/x-phy130027/phd2020/IVP_script/initial_condition_for_3D_%s_shear_flow_fully3D_10picubed_boxsize_MA_eq_%i_theta_in_degrees_eq_%i_u_fluctuatingA_formulation.h5" %(hydro_or_mhd, MA, np.round(theta))
#     else:
#         checkpoint_read      = False
#         ######ICs are imported here.
#         nx_IC                = nx #16 #gridresolution in x-direction in IC
#         ny_IC                = ny #16 #gridresolution in y-direction in IC
#         nz_IC                = nz #16 #gridresolution in z-direction in IC
#         IC_filename_loc      = "/anvil/scratch/x-btripathi/oct2020/%s/post_processing/initial_condition_for_3D_%s_shear_flow_fully3D_10picubed_boxsize_MA_eq_%i_theta_in_degrees_eq_%i_u_fluctuatingA_formulation_FOR_RESTART.h5" %(run_number, hydro_or_mhd, MA, np.round(theta))
# else:
#     IC_filename_loc      = "/anvil/projects/x-phy130027/phd2020/IVP_script/initial_condition_for_3D_%s_shear_flow_fully3D_10picubed_boxsize_u_formulation.h5" %(hydro_or_mhd) 


# #maintain your directories here:
parent_dir = "/anvil/scratch/x-btripathi/"
parent_dir2 = "/anvil/projects/x-phy130027/phd2020/IVP_script"

#maintain your directories here:
#parent_dir = "/Users/bindesh/Downloads/dedalus_test/test"
#parent_dir2= "/Users/bindesh/Downloads/dedalus_test/test"


#==============================================================================================================
#==============================================================================================================



########################################
##Creating directories for each run  ###
########################################
start_time = time.time()

file_dir_1 = '%s/%s' %(month_of_run, run_number)
path_1 = os.path.join(parent_dir, file_dir_1)   
path_2 = '%s/KH_snapshots' %path_1   
path_3 = '%s/post_processing' %path_1
path_4 = '%s/checkpoints' %path_1

if CW.rank == 0:
    access_rights = 0o755

    try:
        os.mkdir(path_1, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_1)
    
    
    
    try:
        os.mkdir(path_2, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_2)
    
    
    
    try:
        os.mkdir(path_3, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_3)
    
    
    
    try:
        os.mkdir(path_4, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_4)
       
    # ###################################################
    # ##Copying files from WORK to SCRATCH in each run###
    # ###################################################
    # files_to_copy = ['%s/MHD_KH_%s.py' %(parent_dir2, script_version_number), '%s/submit_%s.cmd' %(parent_dir2, submit_version_number)]
    # for f1 in files_to_copy:
    #     shutil.copy(f1, '%s' %path_1)

# ##################################
# ##Saving the parameters choosen###
# ##################################
        
# if CW.rank == 0:
#     import json

#     person_dict = { "hydro_or_mhd": hydro_or_mhd,
#     "MA": MA,
#     "Reynolds": Reynolds,          
#     "MagReynolds": MagReynolds,
#     "Schmidt": Schmidt,          
#     "Lx": Lx,
#     "Ly": Ly,
#     "Lz": Lz,          
#     "nx": nx,
#     "ny": ny,
#     "nz": nz,          
#     "sim_time": sim_time,
#     "checkpoint_time": checkpoint_time,
#     "wall_time": wall_time,          
#     "forced": forced,
#     "force_nature": force_nature,
#     "passive_scalar": passive_scalar,  
#     "initcond": initcond, 
#     "checkpoint_read": checkpoint_read,
#     "run_number": run_number,
#     "month_of_run": month_of_run,          
#     "script_version_number": script_version_number,
#     "submit_version_number": submit_version_number,          
#     "a": a,
#     "D_Krook": D_Krook              
#     }


#     filename_param = '%s/KH_parameters.txt' %path_1
#     json_file = open(filename_param, "w")
#     json.dump(person_dict, json_file)
#     json_file.close()
    
#     logger.info('Initial json file created!')



start_init_time = time.time()

############################
##Create bases and domain###
############################

x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', nz, interval=(0, Lz), dealias=3/2)
#domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=mesh)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

kx = domain.elements(0)
ky = domain.elements(1)
kz = domain.elements(2)
k2 = kx**2 + ky**2 + kz**2
k2_mod = k2.copy()
k2_mod[k2 == 0] = 1 #note that k2_mod=1 is carefully chosen to avoid dividing by zero; the whole term ki*kj/k2_mod is any way automatically 0, when kx=ky=kz=0.

############################
##Build problem, BCs     ###
############################
problem_vars = ['ux', 'uy', 'uz', 'Ax', 'Ay', 'Az']

problem                           = de.IVP(domain, variables = problem_vars)
problem.parameters['Re']          = Reynolds
problem.parameters['Rm']          = MagReynolds
problem.parameters['Lx']          = Lx
problem.parameters['Ly']          = Ly
problem.parameters['Lz']          = Lz
problem.parameters['D_Krook']     = D_Krook
problem.parameters['a']           = a
problem.parameters['z1']          = z1
problem.parameters['z2']          = z2

problem.substitutions['u_mean_zprofile']        = "integ(ux, 'x', 'y')/(Lx*Ly)"
problem.substitutions['u_ref_zprofile']         = "tanh((z-z1)/a) - tanh((z-z2)/a) - 1"
problem.substitutions['Laplacian(component)']   = "dx(dx(component))+dy(dy(component))+dz(dz(component))"

if forced == True:
    if force_nature == 'force2':
        problem.substitutions['Fx']   = "-1/Re*Laplacian(u_ref_zprofile) + D_Krook*(u_ref_zprofile - u_mean_zprofile)"
else:
    problem.substitutions['Fx']   = "0.0"

    
problem.substitutions['FL'] = "FractionalLaplacian"
    
#problem.substitutions["curl_x(Cx, Cy, Cz)"]  = "1.0j*ky*Cz - 1.0j*kz*Cy"
#problem.substitutions["curl_y(Cx, Cy, Cz)"]  = "1.0j*kz*Cx - 1.0j*kx*Cz"
#problem.substitutions["curl_z(Cx, Cy, Cz)"]  = "1.0j*kx*Cy - 1.0j*ky*Cx"

problem.substitutions["curl_x(Cx, Cy, Cz)"]  = "dy(Cz) - dz(Cy)"
problem.substitutions["curl_y(Cx, Cy, Cz)"]  = "dz(Cx) - dx(Cz)"
problem.substitutions["curl_z(Cx, Cy, Cz)"]  = "dx(Cy) - dy(Cx)"

problem.substitutions["bx"]             = "curl_x(Ax,Ay,Az)"
problem.substitutions["by"]             = "curl_y(Ax,Ay,Az)"
problem.substitutions["bz"]             = "curl_z(Ax,Ay,Az)"

problem.substitutions["Px"]             = "ux + bx" #P means the Elsasser field Z+ = u + b
problem.substitutions["Py"]             = "uy + by"
problem.substitutions["Pz"]             = "uz + bz"
problem.substitutions["Mx"]             = "ux - bx" #M means the Elsasser field Z- = u - b
problem.substitutions["My"]             = "uy - by"
problem.substitutions["Mz"]             = "uz - bz"

#The following nine nonlinear terms are forward Fourier transformed using the pseudospectral method. 
#These 9 terms are the most expensive in the code as they involve 9 forward FFTs. The other 6 backward FFTs are for u and A.
problem.substitutions["PxMx"]           = "Px*Mx"
problem.substitutions["PxMy"]           = "Px*My"
problem.substitutions["PxMz"]           = "Px*Mz"
problem.substitutions["PyMx"]           = "Py*Mx"
problem.substitutions["PyMy"]           = "Py*My"
problem.substitutions["PyMz"]           = "Py*Mz"
problem.substitutions["PzMx"]           = "Pz*Mx"
problem.substitutions["PzMy"]           = "Pz*My"
problem.substitutions["PzMz"]           = "Pz*Mz"

problem.substitutions["PDotGradM_x"]    = "dx(PxMx) + dy(PyMx) + dz(PzMx)" #x-component of (Z+.grad)Z- = partial_j (P_j M_x)
problem.substitutions["PDotGradM_y"]    = "dx(PxMy) + dy(PyMy) + dz(PzMy)"
problem.substitutions["PDotGradM_z"]    = "dx(PxMz) + dy(PyMz) + dz(PzMz)"

problem.substitutions["MDotGradP_x"]    = "dx(PxMx) + dy(PxMy) + dz(PxMz)" #x-component of (Z-.grad)Z+ = partial_j (P_x M_j)
problem.substitutions["MDotGradP_y"]    = "dx(PyMx) + dy(PyMy) + dz(PyMz)"
problem.substitutions["MDotGradP_z"]    = "dx(PzMx) + dy(PzMy) + dz(PzMz)"

#nonlinearity in u equation is -u.grad u_i + b.grad b_i = 1/2*(- (Z+.grad) Z- - (Z-.grad) Z+) = 1/2*\partial_j( -Pj*Mi - Pi*Mj ).
problem.substitutions["NL_ux"]          = "(-PDotGradM_x-MDotGradP_x)/2"
problem.substitutions["NL_uy"]          = "(-PDotGradM_y-MDotGradP_y)/2"
problem.substitutions["NL_uz"]          = "(-PDotGradM_z-MDotGradP_z)/2"

#nonlinearity in b equation is -u.grad b_i + b.grad u_i = 1/2*( (Z+.grad) Z- - (Z-.grad) Z+) = 1/2*\partial_j( Pj*Mi - Pi*Mj ).
problem.substitutions["NL_bx"]          = "( PDotGradM_x-MDotGradP_x)/2" 
problem.substitutions["NL_by"]          = "( PDotGradM_y-MDotGradP_y)/2"
problem.substitutions["NL_bz"]          = "( PDotGradM_z-MDotGradP_z)/2"

#Note: uxb-grad(Psi) = curl of curl of (uxb)/k^2. This is easily proved: curl of curl of (uxb) = grad(div(uxb)) - Laplacian(uxb) = Laplacian(grad(Psi)) - Laplacian(uxb) = -k^2 * (uxb-grad(Psi)). 
#Where, we have used, div(uxb)=Laplacian(Psi), which is derived by taking divergence of: dt(A)-eta*Laplacian(A) = uxb-grad(Psi). When div(A)=0, div(uxb)=Laplacian(Psi).
problem.substitutions["u_cross_b_minus_grad_Psi_x"] = "curl_x(NL_bx, NL_by, NL_bz)" 
problem.substitutions["u_cross_b_minus_grad_Psi_y"] = "curl_y(NL_bx, NL_by, NL_bz)"  #If these equations are for the mean mode, k2 = 0, then the pressure-correction terms become zero automatically (note that k2_mod=1 is chosen to avoid dividing by zero) 
problem.substitutions["u_cross_b_minus_grad_Psi_z"] = "curl_z(NL_bx, NL_by, NL_bz)" 

problem.substitutions["projected_NL_ux"] = "-Laplacian(NL_ux) + dx(dx(NL_ux)) + dx(dy(NL_uy)) + dx(dz(NL_uz))" #nonlinearity in the ith-component evolution equation is (delta_{i,j} - k_i k_j/k^2)*nonlinearity_j
problem.substitutions["projected_NL_uy"] = "-Laplacian(NL_uy) + dy(dx(NL_ux)) + dy(dy(NL_uy)) + dy(dz(NL_uz))" #nonlinearity in the ith-component evolution equation is (delta_{i,j} - k_i k_j/k^2)*nonlinearity_j
problem.substitutions["projected_NL_uz"] = "-Laplacian(NL_uz) + dz(dx(NL_ux)) + dz(dy(NL_uy)) + dz(dz(NL_uz))" #nonlinearity in the ith-component evolution equation is (delta_{i,j} - k_i k_j/k^2)*nonlinearity_j

############################################################################################
problem.add_equation("dt(Laplacian(ux)) - 1/Re*Laplacian(Laplacian(ux)) = -projected_NL_ux + Laplacian(Fx)", condition="((nx != 0) or (ny != 0)) or (nz != 0)")
problem.add_equation("dt(Laplacian(uy)) - 1/Re*Laplacian(Laplacian(uy)) = -projected_NL_uy                ", condition="((nx != 0) or (ny != 0)) or (nz != 0)")
problem.add_equation("dt(Laplacian(uz)) - 1/Re*Laplacian(Laplacian(uz)) = -projected_NL_uz                ", condition="((nx != 0) or (ny != 0)) or (nz != 0)")
problem.add_equation("dt(Laplacian(Ax)) - 1/Rm*Laplacian(Laplacian(Ax)) = -u_cross_b_minus_grad_Psi_x  ",    condition="((nx != 0) or (ny != 0)) or (nz != 0)")
problem.add_equation("dt(Laplacian(Ay)) - 1/Rm*Laplacian(Laplacian(Ay)) = -u_cross_b_minus_grad_Psi_y  ",    condition="((nx != 0) or (ny != 0)) or (nz != 0)")
problem.add_equation("dt(Laplacian(Az)) - 1/Rm*Laplacian(Laplacian(Az)) = -u_cross_b_minus_grad_Psi_z  ",    condition="((nx != 0) or (ny != 0)) or (nz != 0)")

#If the following equations are for the mean mode, kx=ky=kz=k2=0. Under such a scenario, the pressure-correction terms become zero automatically (note that k2_mod=1 is carefully chosen to avoid dividing by zero; the whole term ki*kj/k2_mod = 0, when kx=ky=kz=0.) 
#I think this means: When kx=ky=kz=0, there is neither global constant, uniform motion nor a global uniform, constant field—because the domain is triply periodic, the entire infinite space is considered. Since monopoles do not exist (div(b)=0), the whole system averaged field in the infinite universe is zero.
problem.add_equation("ux = 0", condition="((nx == 0) and (ny == 0)) and (nz == 0)")
problem.add_equation("uy = 0", condition="((nx == 0) and (ny == 0)) and (nz == 0)")
problem.add_equation("uz = 0", condition="((nx == 0) and (ny == 0)) and (nz == 0)")
problem.add_equation("Ax = 0", condition="((nx == 0) and (ny == 0)) and (nz == 0)")
problem.add_equation("Ay = 0", condition="((nx == 0) and (ny == 0)) and (nz == 0)")
problem.add_equation("Az = 0", condition="((nx == 0) and (ny == 0)) and (nz == 0)")
############################################################################################


# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')


#######################
##Initial conditions###
####################### 
#Solver state
ux   = solver.state['ux']
uy   = solver.state['uy']
uz   = solver.state['uz']
Ax   = solver.state['Ax']
Ay   = solver.state['Ay']
Az   = solver.state['Az']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1) #This for convergence check studies, which should all have at least (16,16, 512) Fourier modes. Higher resolution simulations.
slices = domain.dist.grid_layout.slices(scales=1)

#rand = np.random.RandomState(seed=23)
#noise = rand.standard_normal(gshape)[slices]
#uz['g'] = 1e-8 * noise 
'''
for kin1 in range(0,5):
    for kin2 in range(-4,5):
        if (kin1!=0):
            ux['g'] += 1e-8*np.cos(2*np.pi/Ly*kin1*y+2*np.pi/Lz*kin2*z)
            uy['g'] += 1e-8*np.cos(2*np.pi/Lz*kin1*z+2*np.pi/Lz*kin2*x)
            uz['g'] += 1e-8*np.cos(2*np.pi/Lx*kin1*x+2*np.pi/Lx*kin2*y)
        elif ((kin1==0) and (kin2>0)):
            ux['g'] += 1e-8*np.cos(2*np.pi/Ly*kin1*y+2*np.pi/Lz*kin2*z)
            uy['g'] += 1e-8*np.cos(2*np.pi/Lz*kin1*z+2*np.pi/Lz*kin2*x)
            uz['g'] += 1e-8*np.cos(2*np.pi/Lx*kin1*x+2*np.pi/Lx*kin2*y)
'''
'''
#Either purely kx=0, or ky=0, or kz=0 mode is excited in the next FOR loop. Hermiticity demands summation over postive ki ONLY.
#The following ICs satisfy divu =0 and are exact nonlinear solutions to the Euler equation.
#The ICs are of the form: v = amp*cos(k1*x+k2*y*k3*z+phi_random), where either k1, or k2, or k3 is zero for a given wavenumber (k1,k2,k3).
#phi_random is a random phase, lying between (-pi,pi), for every wavenumber.
for ki in range(1,5):
    ux['g'] += 1e-8*np.cos(2*np.pi/Ly*0*y + 2*np.pi/Lz*ki*z + ki/4*(-67)/100*3.1415926535897932)
    uy['g'] += 1e-8*np.cos(2*np.pi/Lz*0*z + 2*np.pi/Lz*ki*x + ki/4*(7)/100*3.1415926535897932)
    uz['g'] += 1e-8*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Lx*ki*y + ki/4*(-71)/100*3.1415926535897932)

#Two wavenumebers (among kx, ky, and kz) can be non-zero now. When ki=0, hermicity is to be imposed, but the following equations satisfy it, because
#I have written separate DO loops for separate non-negative values of one of the wavenumbers: 0, 1, 2, 3, and 4.
#Since cosine is used, no negative values (-1,-2,-3, and -4) are needed for that wavenumber—due to Hermiticity.
for ki in range(-4,5):
    ux['g'] += 1e-8*np.cos(2*np.pi/Ly*1*y + 2*np.pi/Lz*ki*z + ki/4*(61)/100*3.1415926535897932)
    uy['g'] += 1e-8*np.cos(2*np.pi/Lz*1*z + 2*np.pi/Lz*ki*x + ki/4*(-44)/100*3.1415926535897932)
    uz['g'] += 1e-8*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Lx*ki*y + ki/4*(0)/100*3.1415926535897932)
for ki in range(-4,5):
    ux['g'] += 1e-8*np.cos(2*np.pi/Ly*2*y + 2*np.pi/Lz*ki*z + ki/4*(37)/100*3.1415926535897932)
    uy['g'] += 1e-8*np.cos(2*np.pi/Lz*2*z + 2*np.pi/Lz*ki*x + ki/4*(18)/100*3.1415926535897932)
    uz['g'] += 1e-8*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Lx*ki*y + ki/4*(31)/100*3.1415926535897932)
for ki in range(-4,5):
    ux['g'] += 1e-8*np.cos(2*np.pi/Ly*3*y + 2*np.pi/Lz*ki*z + ki/4*(6)/100*3.1415926535897932)
    uy['g'] += 1e-8*np.cos(2*np.pi/Lz*3*z + 2*np.pi/Lz*ki*x + ki/4*(-79)/100*3.1415926535897932)
    uz['g'] += 1e-8*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Lx*ki*y + ki/4*(-20)/100*3.1415926535897932)
for ki in range(-4,5):
    ux['g'] += 1e-8*np.cos(2*np.pi/Ly*4*y + 2*np.pi/Lz*ki*z + ki/4*(49)/100*3.1415926535897932)
    uy['g'] += 1e-8*np.cos(2*np.pi/Lz*4*z + 2*np.pi/Lz*ki*x + ki/4*(-80)/100*3.1415926535897932)
    uz['g'] += 1e-8*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Lx*ki*y + ki/4*(84)/100*3.1415926535897932)
'''
Cx = domain.new_field()
Cy = domain.new_field()
Cz = domain.new_field()
Cx.set_scales(1, keep_data=False)
Cy.set_scales(1, keep_data=False)
Cz.set_scales(1, keep_data=False)

# !Hermiticity demands summation over postive ki (positive n) only if m=0.
# !The ICs are of the form: v = amp/sqrt(m**2+n**2)*cos(m*x+n*y+phi_random)*(exp(-((z-z1)/sigma)**2)+exp(-((z-z2)/sigma)**2)).
# !For every horizontal wavenumber (m,n), phi_random is a random phase, lying between (-pi,pi).
# !By hand, I am prescribing below random numbers from a random number generator.
for ki in range(1,5):
    Cx['g'] += amp/np.sqrt(0**2+ki**2)*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Ly*ki*y + ki/4*(-67)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cy['g'] += amp/np.sqrt(0**2+ki**2)*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Ly*ki*y + ki/4*(7)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cz['g'] += amp/np.sqrt(0**2+ki**2)*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Ly*ki*y + ki/4*(-71)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))

# !When ki=0, hermicity is to be imposed, but the following equations satisfy it, because
# !I have written separate DO loops for separate non-negative values of one of the wavenumbers: 0, 1, 2, 3, and 4.
# !Since cosine is used, no negative values (-1,-2,-3, and -4) are needed for that wavenumber, due to Hermiticity.
for ki in range(-4,5):
    Cx['g'] += amp/np.sqrt(1**2+ki**2)*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Ly*ki*y + ki/4*(61)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cy['g'] += amp/np.sqrt(1**2+ki**2)*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Ly*ki*y + ki/4*(-44)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cz['g'] += amp/np.sqrt(1**2+ki**2)*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Ly*ki*y + ki/4*(0)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
for ki in range(-4,5):
    Cx['g'] += amp/np.sqrt(2**2+ki**2)*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Ly*ki*y + ki/4*(37)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cy['g'] += amp/np.sqrt(2**2+ki**2)*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Ly*ki*y + ki/4*(18)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cz['g'] += amp/np.sqrt(2**2+ki**2)*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Ly*ki*y + ki/4*(31)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
for ki in range(-4,5):
    Cx['g'] += amp/np.sqrt(3**2+ki**2)*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Ly*ki*y + ki/4*(6)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cy['g'] += amp/np.sqrt(3**2+ki**2)*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Ly*ki*y + ki/4*(-79)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cz['g'] += amp/np.sqrt(3**2+ki**2)*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Ly*ki*y + ki/4*(-20)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
for ki in range(-4,5):
    Cx['g'] += amp/np.sqrt(4**2+ki**2)*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Ly*ki*y + ki/4*(49)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cy['g'] += amp/np.sqrt(4**2+ki**2)*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Ly*ki*y + ki/4*(-80)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Cz['g'] += amp/np.sqrt(4**2+ki**2)*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Ly*ki*y + ki/4*(84)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))

ux['c'] = Cz.differentiate(y=1)['c'] - Cy.differentiate(z=1)['c']
uy['c'] = Cx.differentiate(z=1)['c'] - Cz.differentiate(x=1)['c']
uz['c'] = Cy.differentiate(x=1)['c'] - Cx.differentiate(y=1)['c']

#uz['c'][(kx==0).flatten(), (ky==0).flatten()] *= 0 #removing the mean velocity perturbations
temp_data = domain.new_field()
temp_data2 = domain.new_field()
temp_data.set_scales(1, keep_data=False)
temp_data2.set_scales(1, keep_data=False)
#temp_data['c'][(kx!=0).flatten()] = uz.differentiate(z=1)['c'][(kx!=0).flatten()] * 1/(-1.0j) * kx[(kx!=0).flatten()]**(-1)    
#temp_data2['c'][(kx==0).flatten(), (ky!=0).flatten()] = uz.differentiate(z=1)['c'][(kx==0).flatten(), (ky!=0).flatten()] * 1/(-1.0j) * ky[0, (ky!=0).flatten()]**(-1)                            
ux['g'] += (np.tanh((z-z1)/a) - np.tanh((z-z2)/a) - np.cos(z*0))

#rand = np.random.RandomState(seed=21)
#noise = rand.standard_normal(gshape)[slices]
#Az['g'] = 1e-8 * noise 
'''
for kin in range(1,4):
    Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*kin*y) + 1e-8*np.sin(2*np.pi/Lz*kin*z)
    Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*kin*z) + 1e-8*np.sin(2*np.pi/Lx*kin*x)
    Az['g'] += 1e-8*np.cos(2*np.pi/Lx*kin*x) + 1e-8*np.sin(2*np.pi/Ly*kin*y)
'''
'''
for kin1 in range(0,5):
    for kin2 in range(-4,5):
        if (kin1!=0):
            Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*kin1*y+2*np.pi/Lz*kin2*z)
            Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*kin1*z+2*np.pi/Lz*kin2*x)
            Az['g'] += 1e-8*np.cos(2*np.pi/Lx*kin1*x+2*np.pi/Lx*kin2*y)
        elif ((kin1==0) and (kin2>0)):
            Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*kin1*y+2*np.pi/Lz*kin2*z)
            Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*kin1*z+2*np.pi/Lz*kin2*x)
            Az['g'] += 1e-8*np.cos(2*np.pi/Lx*kin1*x+2*np.pi/Lx*kin2*y)
'''
'''
#Either purely kx=0, or ky=0, or kz=0 mode is excited in the next FOR loop. Hermiticity demands summation over postive ki only. 
#The following ICs satisfy div a =0. Together with the ICs of the flow, the following ICs are exact nonlinear solutions to the ideal MHD equations.
#The ICs are of the form: A = amp*cos(k1*x+k2*y*k3*z+phi_random), where either k1, or k2, or k3 is zero for a given wavenumber (k1,k2,k3). 
#phi_random is a random phase, lying between (-pi,pi), for every wavenumber.
for ki in range(1,5):
    Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*0*y + 2*np.pi/Lz*ki*z + ki/4*(50)/100*3.1415926535897932)
    Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*0*z + 2*np.pi/Lz*ki*x + ki/4*(-42)/100*3.1415926535897932)
    Az['g'] += 1e-8*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Lx*ki*y + ki/4*(-81)/100*3.1415926535897932)

#Two wavenumebers (among kx, ky, and kz) can be non-zero now. When ki=0, hermicity is to be imposed, but the following equations satisfy it, because
#I have written separate DO loops for separate non-negative values of one of the wavenumbers: 0, 1, 2, 3, and 4. 
#Since cosine is used, no negative values (-1,-2,-3, and -4) are needed for that wavenumber—due to Hermiticity.
for ki in range(-4,5):
    Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*1*y + 2*np.pi/Lz*ki*z + ki/4*(56)/100*3.1415926535897932)
    Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*1*z + 2*np.pi/Lz*ki*x + ki/4*(85)/100*3.1415926535897932)
    Az['g'] += 1e-8*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Lx*ki*y + ki/4*(-50)/100*3.1415926535897932)
for ki in range(-4,5):
    Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*2*y + 2*np.pi/Lz*ki*z + ki/4*(-26)/100*3.1415926535897932)
    Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*2*z + 2*np.pi/Lz*ki*x + ki/4*(-1)/100*3.1415926535897932)
    Az['g'] += 1e-8*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Lx*ki*y + ki/4*(-24)/100*3.1415926535897932)
for ki in range(-4,5):
    Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*3*y + 2*np.pi/Lz*ki*z + ki/4*(84)/100*3.1415926535897932)
    Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*3*z + 2*np.pi/Lz*ki*x + ki/4*(18)/100*3.1415926535897932)
    Az['g'] += 1e-8*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Lx*ki*y + ki/4*(-21)/100*3.1415926535897932)
for ki in range(-4,5):
    Ax['g'] += 1e-8*np.cos(2*np.pi/Ly*4*y + 2*np.pi/Lz*ki*z + ki/4*(25)/100*3.1415926535897932)
    Ay['g'] += 1e-8*np.cos(2*np.pi/Lz*4*z + 2*np.pi/Lz*ki*x + ki/4*(42)/100*3.1415926535897932)
    Az['g'] += 1e-8*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Lx*ki*y + ki/4*(20)/100*3.1415926535897932)
'''

Dx = domain.new_field()
Dy = domain.new_field()
Dz = domain.new_field()
Dx.set_scales(1, keep_data=False)
Dy.set_scales(1, keep_data=False)
Dz.set_scales(1, keep_data=False)

# !Hermiticity demands summation over postive ki (positive n) only if m=0.
# !The ICs are of the form: b = curl of A,
#where A = amp/(m**2+n**2)*cos(m*x+n*y+phi_random)*(exp(-((z-z1)/sigma)**2)+exp(-((z-z2)/sigma)**2)).
# !For every horizontal wavenumber (m,n), phi_random is a random phase, lying between (-pi,pi).
# !By hand, I am prescribing below random numbers from a random number generator.
for ki in range(1,5):
    Dx['g'] += amp/(0**2+ki**2)*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Ly*ki*y + ki/4*(50)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dy['g'] += amp/(0**2+ki**2)*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Ly*ki*y + ki/4*(-42)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dz['g'] += amp/(0**2+ki**2)*np.cos(2*np.pi/Lx*0*x + 2*np.pi/Ly*ki*y + ki/4*(-81)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))

# !When ki=0, hermicity is to be imposed, but the following equations satisfy it, because
# !I have written separate DO loops for separate non-negative values of one of the wavenumbers: 0, 1, 2, 3, and 4.
# !Since cosine is used, no negative values (-1,-2,-3, and -4) are needed for that wavenumber, due to Hermiticity.
for ki in range(-4,5):
    Dx['g'] += amp/(1**2+ki**2)*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Ly*ki*y + ki/4*(56)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dy['g'] += amp/(1**2+ki**2)*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Ly*ki*y + ki/4*(85)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dz['g'] += amp/(1**2+ki**2)*np.cos(2*np.pi/Lx*1*x + 2*np.pi/Ly*ki*y + ki/4*(-50)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
for ki in range(-4,5):
    Dx['g'] += amp/(2**2+ki**2)*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Ly*ki*y + ki/4*(-26)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dy['g'] += amp/(2**2+ki**2)*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Ly*ki*y + ki/4*(-1)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dz['g'] += amp/(2**2+ki**2)*np.cos(2*np.pi/Lx*2*x + 2*np.pi/Ly*ki*y + ki/4*(-24)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
for ki in range(-4,5):
    Dx['g'] += amp/(3**2+ki**2)*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Ly*ki*y + ki/4*(84)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dy['g'] += amp/(3**2+ki**2)*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Ly*ki*y + ki/4*(18)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dz['g'] += amp/(3**2+ki**2)*np.cos(2*np.pi/Lx*3*x + 2*np.pi/Ly*ki*y + ki/4*(-21)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
for ki in range(-4,5):
    Dx['g'] += amp/(4**2+ki**2)*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Ly*ki*y + ki/4*(25)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dy['g'] += amp/(4**2+ki**2)*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Ly*ki*y + ki/4*(42)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))
    Dz['g'] += amp/(4**2+ki**2)*np.cos(2*np.pi/Lx*4*x + 2*np.pi/Ly*ki*y + ki/4*(20)/100*3.1415926535897932)*(np.exp(-((z-z1)/sigma)**2)+np.exp(-((z-z2)/sigma)**2))

Ax['c'] = Dz.differentiate(y=1)['c'] - Dy.differentiate(z=1)['c']
Ay['c'] = Dx.differentiate(z=1)['c'] - Dz.differentiate(x=1)['c']
Az['c'] = Dy.differentiate(x=1)['c'] - Dx.differentiate(y=1)['c']

#Az['c'][(kx==0).flatten(), (ky==0).flatten()] *= 0 #removing the mean magnetic perturbations
temp_data = domain.new_field()
temp_data2 = domain.new_field()
temp_data.set_scales(1, keep_data=False)
temp_data2.set_scales(1, keep_data=False)
#temp_data['c'][(kx!=0).flatten()] = Az.differentiate(z=1)['c'][(kx!=0).flatten()] * 1/(-1.0j) * kx[(kx!=0).flatten()]**(-1)    
#temp_data2['c'][(kx==0).flatten(), (ky!=0).flatten()] = Az.differentiate(z=1)['c'][(kx==0).flatten(), (ky!=0).flatten()] * 1/(-1.0j) * ky[0, (ky!=0).flatten()]**(-1)                            
#Ax['g'] = temp_data['g']
#Ay['g'] = temp_data2['g']

# Integration parameters
logger.info('Initial conditions have been prepared. Proceeding next!')

dt0 = 0.001*Lx/nx #This st0 has been used in cfl condn. max_dt = 100*dt0
dt = dt0    #for initial_dt = dt in CFL

#____________________________________________________________________________________________________________________________________________________________________________
if checkpoint_read == True:
    logger.info('Reading checkpoint data for initializing the continued-integration')
    dt0 = 0.01*Lx/nx #This st0 has been used in cfl condn. max_dt = 100*dt0
    # Restart
    checkpoint_data = '%s/checkpoints_s1.h5' %path_4
    index = -1
        #chpt_format='g'
        
    with h5py.File(checkpoint_data, mode='r') as file:
        # Load solver attributes
        write = file['scales']['write_number'][index]
        try:
            dt = file['scales']['timestep'][index]
        except KeyError:
            dt = None
        solver.iteration = solver.initial_iteration = file['scales']['iteration'][index]
        solver.sim_time  = solver.initial_sim_time  = file['scales']['sim_time'][index]
        # Log restart info
        logger.info("Loading iteration: {}".format(solver.iteration))
        logger.info("Loading write: {}".format(write))
        logger.info("Loading sim time: {}".format(solver.sim_time))
        logger.info("Loading timestep: {}".format(dt))
        # Load fields
        for field in solver.state.fields:
            dset = file['tasks'][field.name]
            # Find matching layout
            for layout in solver.domain.dist.layouts:
                if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                    break
            else:
                raise ValueError("No matching layout")
            # Set scales to match saved data
            scales = dset.shape[1:] / layout.global_shape(scales=1)
            scales[~layout.grid_space] = 1
            # Extract local data from global dset
            dset_slices = (index,) + layout.slices(tuple(scales))
            local_dset = dset[dset_slices]
            # Copy to field
            field_slices = tuple(slice(n) for n in local_dset.shape)
            field.set_scales(scales, keep_data=False)
            field[layout][field_slices] = local_dset
            field.set_scales(solver.domain.dealias, keep_data=True)
                
#____________________________________________________________________________________________________________________________________________________________________________

    
###################
solver.stop_sim_time  = sim_time 
solver.stop_wall_time = wall_time 
solver.stop_iteration = max_steps

cfl        = flow_tools.CFL(solver, initial_dt=dt, cadence = 100, safety = 0.1, max_change = 1.5, max_dt = 100*dt0, threshold = 0.05) 
#I've played with changing cadence from 1 to 200, and I see dt often not changing even after >500 iterations once the nonlinear phase is reached.

cfl.add_velocities(('ux', 'uy', 'uz'))
cfl.add_velocities(('bx', 'by', 'bz')) #MA is not needed (because I normalize everythng with respect to U_0). #cfl.add_velocities(('Bx','Bz')) with 4*pi*rho = 1 is the same as Bx/MA for CFL conditions.



#Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property("1/2* ((ux*ux + uy*uy + uz*uz) + (bx*bx + by*by + bz*bz)) ",       name='energy_blow_up_check')

###################
##Analysis tasks###
###################
analysis = solver.evaluator.add_file_handler('%s' %path_2, sim_dt = 2, mode = 'append') 
# #analysis.add_system(solver.state, layout='g')
analysis.add_task('ux',  name='ux', layout='g')
analysis.add_task('uy',  name='uy', layout='g')
analysis.add_task('uz',  name='uz', layout='g')
analysis.add_task('Ax',  name='Ax', layout='g')
analysis.add_task('Ay',  name='Ay', layout='g')
analysis.add_task('Az',  name='Az', layout='g')


# # Analysis
# snap = solver.evaluator.add_file_handler('snapshots', sim_dt=0.2, max_writes=10)
# snap.add_task("interp(p, z=0)", scales=1, name='p midplane')
# snap.add_task("interp(b, z=0)", scales=1, name='b midplane')
# snap.add_task("interp(u, z=0)", scales=1, name='u midplane')
# snap.add_task("interp(v, z=0)", scales=1, name='v midplane')
# snap.add_task("interp(w, z=0)", scales=1, name='w midplane')
# snap.add_task("integ(b, 'z')", name='b integral x4', scales=4)


#This is only to continue the integration with restart.h5 file
final_chpt = solver.evaluator.add_file_handler('%s' %path_4, sim_dt = checkpoint_time, mode = 'overwrite') # sim_dt = sim_time-1 is because I have tried checking running the code and it turns  out that dedalus records data always t = actual_time-1 as the first frame is missed out somehow.
final_chpt.add_system(solver.state)



def plotting_func(count_t, plot_counter):
    #import seaborn as sns
    #sns.set()
    #sns.set_style("ticks")

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib  import cm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import LogFormatter

    color_cmap = 'RdBu' #'inferno'

    lw = 0.7

    fnt = 20
    ms = 3

    nrows = 1
    ncols = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18*0.65,12*0.65)) #plots an asthetically pleasing graph using golden/divine ratio

    

    #####################
    ux.set_scales(1)
    uxmean_instantaneous = domain.new_field()
    uxmean_instantaneous.set_scales(1, keep_data=False)
    uxmean_instantaneous['c'][0, 0, :] = ux['c'][0, 0, :]

    zindx = int(nz/4*3)
    data = ux['g'][:, :, zindx] -  uxmean_instantaneous['g'][:, :, zindx]
    vmax_or_vmin = np.max(np.abs(data))
    
    ii = 0
    colorplot = ax[ii].pcolormesh(x.flatten(), y.flatten(), data.T, cmap='RdBu_r', vmax = vmax_or_vmin, vmin = -vmax_or_vmin, zorder=0)
    ax[ii].set_xlabel(r"$x$", fontsize=fnt)
    ax[ii].set_ylabel(r"$y$", fontsize=fnt)
    ax[ii].tick_params(axis = 'both', which = 'both', direction = 'in', labelsize=fnt, top=True, right=True, bottom=True, left=True)
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    ax[ii].xaxis.set_minor_locator(MultipleLocator(1))
    ax[ii].yaxis.set_minor_locator(MultipleLocator(1))
    ax[ii].annotate(r"$u_x$",  (0.4, 1.07), xycoords='axes fraction', va='center', fontsize=fnt, color='black')
    fig.colorbar(colorplot)
    
    #####################
    uz.set_scales(1)
    data = uz['g'][:, 0, :]
    vmax_or_vmin = np.max(np.abs(data))
    
    ii = 1
    colorplot = ax[ii].pcolormesh(x.flatten(), z[0][0], data.T, cmap='RdBu_r', vmax = vmax_or_vmin, vmin = -vmax_or_vmin, zorder=0)
    ax[ii].set_xlabel(r"$x$", fontsize=fnt)
    ax[ii].set_ylabel(r"$z$", fontsize=fnt)
    ax[ii].tick_params(axis = 'both', which = 'both', direction = 'in', labelsize=fnt, top=True, right=True, bottom=True, left=True)
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    ax[ii].xaxis.set_minor_locator(MultipleLocator(1))
    ax[ii].yaxis.set_minor_locator(MultipleLocator(1))
    ax[ii].annotate(r"$u_z$",  (0.4, 1.07), xycoords='axes fraction', va='center', fontsize=fnt, color='black')
    ax[ii].annotate(r"$t=%e$" %count_t,  (-0.6, 1.17), xycoords='axes fraction', va='center', fontsize=fnt, color='black')
    fig.colorbar(colorplot)
    
    plt.show()
    #fig.savefig('/home/x-btripathi/ALL_work_on_ONDEMAND/3D_MHD/analysis/plots/2024/jan2024/wk1/plot1b.png', dpi=1200, bbox_inches='tight', pad_inches=0.05)

    #fig.savefig('/home/x-btripathi/ALL_work_on_ONDEMAND/3D_MHD/analysis/plots/2025/mar2025/wk1/test_run2/test_%i.png' %plot_counter, dpi=300, bbox_inches='tight', pad_inches=0.1)
    

###################
##Actual Solver ###
###################
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    plot_counter  = 0
    while solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt)
        if solver.iteration % 100 == 0:
            logger.info('Iteration: %i, Time: %e, max_energy: %f, dt: %e' %(solver.iteration, solver.sim_time, flow.max('energy_blow_up_check'), dt))
            #plotting_func(solver.sim_time, plot_counter)
            plot_counter += 1
        if dt<10**(-10):
            solver.ok=False
            logger.info('dt less than 10**(-10) reached!')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()        
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Re: %i' %Reynolds)
    solver.log_stats()
    
    
    
#     ##################################
#     ##Saving the parameters choosen###
#     ##################################
        
#     if checkpoint_read == False:
#         import json

#         person_dict = {"Iterations": solver.iteration,
#         "Sim end time": solver.sim_time,
#         "Run time": (end_time - start_time), 
#         "hydro_or_mhd": hydro_or_mhd,
#         "MA": MA,
#         "Reynolds": Reynolds,          
#         "MagReynolds": MagReynolds,
#         "Schmidt": Schmidt,          
#         "Lx": Lx,
#         "Ly": Ly,
#         "Lz": Lz,          
#         "nx": nx,
#         "ny": ny,
#         "nz": nz,          
#         "sim_time": sim_time,
#         "checkpoint_time": checkpoint_time,
#         "wall_time": wall_time,          
#         "forced": forced,
#         "force_nature": force_nature,
#         "passive_scalar": passive_scalar,  
#         "initcond": initcond, 
#         "checkpoint_read": checkpoint_read,
#         "run_number": run_number,
#         "month_of_run": month_of_run,          
#         "script_version_number": script_version_number,
#         "submit_version_number": submit_version_number,          
#         "a": a,
#         "D_Krook": D_Krook              
#         }


#         filename_param = '%s/KH_parameters.txt' %path_1
#         json_file = open(filename_param, "w")
#         json.dump(person_dict, json_file)
#         json_file.close()
    
#     logger.info('The simulation has ended! Merging files now!')
    
    
    
    
    
    
    ########################################
    ##Merging files from mpi parallel run###
    ########################################
    
    from dedalus.tools import post
    post.merge_process_files("%s" %path_2, cleanup=True)
    post.merge_process_files("%s" %path_4, cleanup=True)



logger.info('Final check! Yaay! You are amazing!')

