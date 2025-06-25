"""
Dedalus script for 3D hydro or MHD KH instability with a Krook forcing.

This script uses Fourier bases in the x,y-directions.
The z-axis is decomposed in to Chebyshev polynomials. 

The flow is directed along the positive x-axis in z>0 domain and along the negative x-axis 
in z<0 domain. ICs are carefully prepared separately in lab (K41), by inverting the Laplacian 
to find a self-consistent pressure and Psi (in uncurled magnetic induction equation) 
so that $p$ and $\Psi$ satisfy the solenoidal property of the flow and the vector potential A. 
Then, such ICs are imported here.

                  z--------->
                   ---------> U_0(z)
                   --------->
                   -->
                <--
           <-------
           <-------
           <-------

UPDATE:
All magnetic variables are decomposed into background and time-evolving parts, 
where the background part refers to the background vector potenial A_b such that 
curl of A_b = (B_0 costheta, B_0 sintheta, 0). 
When the background magnetic field is uniform in space, the vector potential A_b becomes a linear function of spatial coordinates, 
which is numerically worrisome to expand in a finite number of periodic basis functions. So, I have now separated
the magnetic variables into background and time-evolving parts.
The flow, however, is solved without doing such a decomposition, as it is not necessary.
 
 
---Bindesh Tripathi
June 2023.

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




#==============================================================================================================
#==============================================================================================================
#CHOOSE YOUR PARAMETERS
#==============================================================================================================
#==============================================================================================================
hydro_or_mhd         = 'mhd' #'hydro', 'mhd'
MA                   = 30.0      #40 #2.5 #Alfven Mach number, defined as U_0/v_A
theta                = 30.0 #30/180*np.pi #degrees to radians
Reynolds             = 50 
MagReynolds          = 50
D_Krook              = 2.0       #0 #0.1  
Schmidt              = 1.0
Lx                   = 10.0*np.pi
Ly                   = 10.0*np.pi
Lz                   = 10.0*np.pi
nx                   = 128  #512 #96 #256*2 #1024  #192
ny                   = 128     #512 #96 #256*2 #1024  #192
nz                   = 512  #512 #96 #1024  #256 #96
mesh                 = [32, 64] #no. of processros along x and y directions. product of this should match with number of total processir asked.   
sim_time             = 50001     #301 #82.00 #148.00 #65.55 #45.01 #make this 1 more than a perfect even number so that  sim_time = integer*checkpoint_time +1.
checkpoint_time      = 20
wall_time            = 86400/24*90 #The last one hour out of 48 hours would be devoted to merge files at the end of solver calculations.
max_steps            = np.inf 
forced               = True   #False #True
force_nature         = 'force2'
passive_scalar       = False    #False #True
initcond             = 'noise' #'coherent'  #'noise'
checkpoint_read      = True
run_number           = 'run306'  #run1 #used to store files for separate runs in separate directory
month_of_run         = 'oct2020'
script_version_number= 'run306' # change script title as well with the change in version.
submit_version_number= script_version_number     # change submit.cmd file as well with the change in version. 
                                                 # This feature is added to allow multiple queue jobs in sbatch Stampede2


a                    = 1

restart_after_core_crash = False




######ICs are imported here.
if restart_after_core_crash == False:
    nx_IC                = 16 #gridresolution in x-direction in IC
    ny_IC                = 16 #gridresolution in y-direction in IC
    nz_IC                = 512 #gridresolution in z-direction in IC


#Finding where the IC data is located.
if hydro_or_mhd == 'mhd':
    if restart_after_core_crash == False:
        IC_filename_loc      = "/anvil/projects/x-phy130027/phd2020/IVP_script/initial_condition_for_3D_%s_shear_flow_fully3D_10picubed_boxsize_MA_eq_%i_theta_in_degrees_eq_%i_u_fluctuatingA_formulation.h5" %(hydro_or_mhd, MA, np.round(theta))
    else:
        checkpoint_read      = False
        ######ICs are imported here.
        nx_IC                = nx #16 #gridresolution in x-direction in IC
        ny_IC                = ny #16 #gridresolution in y-direction in IC
        nz_IC                = nz #16 #gridresolution in z-direction in IC
        IC_filename_loc      = "/anvil/scratch/x-btripathi/oct2020/%s/post_processing/initial_condition_for_3D_%s_shear_flow_fully3D_10picubed_boxsize_MA_eq_%i_theta_in_degrees_eq_%i_u_fluctuatingA_formulation_FOR_RESTART.h5" %(run_number, hydro_or_mhd, MA, np.round(theta))
else:
    IC_filename_loc      = "/anvil/projects/x-phy130027/phd2020/IVP_script/initial_condition_for_3D_%s_shear_flow_fully3D_10picubed_boxsize_u_formulation.h5" %(hydro_or_mhd) 


#maintain your directories here:
parent_dir = "/anvil/scratch/x-btripathi/"
parent_dir2 = "/anvil/projects/x-phy130027/phd2020/IVP_script"


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
       
    ###################################################
    ##Copying files from WORK to SCRATCH in each run###
    ###################################################
    files_to_copy = ['%s/MHD_KH_%s.py' %(parent_dir2, script_version_number), '%s/submit_%s.cmd' %(parent_dir2, submit_version_number)]
    for f1 in files_to_copy:
        shutil.copy(f1, '%s' %path_1)

##################################
##Saving the parameters choosen###
##################################
        
if CW.rank == 0:
    import json

    person_dict = { "hydro_or_mhd": hydro_or_mhd,
    "theta": theta,
    "MA": MA,
    "Reynolds": Reynolds,          
    "MagReynolds": MagReynolds,
    "Schmidt": Schmidt,          
    "Lx": Lx,
    "Ly": Ly,
    "Lz": Lz,          
    "nx": nx,
    "ny": ny,
    "nz": nz,          
    "sim_time": sim_time,
    "checkpoint_time": checkpoint_time,
    "wall_time": wall_time,          
    "forced": forced,
    "force_nature": force_nature,
    "passive_scalar": passive_scalar,  
    "initcond": initcond, 
    "checkpoint_read": checkpoint_read,
    "run_number": run_number,
    "month_of_run": month_of_run,          
    "script_version_number": script_version_number,
    "submit_version_number": submit_version_number,          
    "a": a,
    "D_Krook": D_Krook              
    }


    filename_param = '%s/KH_parameters.txt' %path_1
    json_file = open(filename_param, "w")
    json.dump(person_dict, json_file)
    json_file.close()
    
    logger.info('Initial json file created!')

    

############################
##Create bases and domain###
############################

x_basis = de.Fourier('x',   nx, interval = (0, Lx),       dealias = 3/2)
y_basis = de.Fourier('y',   ny, interval = (0, Ly),       dealias = 3/2)
z_basis = de.Chebyshev('z', nz, interval = (-Lz/2, Lz/2), dealias = 3/2)
domain  = de.Domain([x_basis, y_basis, z_basis], grid_dtype = np.float64, mesh=mesh)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)


###############################################
#Reading the IC data for ux, uy, uz, p.

temp_ux = domain.new_field()
temp_uy = domain.new_field()
temp_uz = domain.new_field()
temp_p  = domain.new_field()

if hydro_or_mhd == 'mhd':
    temp_Ax = domain.new_field()
    temp_Ay = domain.new_field()
    temp_Az = domain.new_field()
    temp_Psi = domain.new_field()

gslices = domain.dist.grid_layout.slices(scales=1) #global slices to be passed to each processor for loading the inputs_data.
gslices_one_res_to_other_res = domain.dist.grid_layout.slices(scales=(nx_IC/nx, ny_IC/ny, nz_IC/nz)) #global slices to be passed to each processor for loading the inputs_data.

logger.info("IC data loading has begun.")

inputs = h5py.File('%s' %(IC_filename_loc), 'r')

temp_ux.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)
temp_uy.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)
temp_uz.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)
temp_p.set_scales( (nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)

if hydro_or_mhd == 'mhd':
    temp_Ax.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)
    temp_Ay.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)
    temp_Az.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)
    temp_Psi.set_scales((nx_IC/nx, ny_IC/ny, nz_IC/nz), keep_data=False)

inputs_data  = inputs['ux/ux/ux']
temp_ux['g'] = inputs_data[:, :, :][gslices_one_res_to_other_res]

inputs_data  = inputs['uy/uy/uy']
temp_uy['g'] = inputs_data[:, :, :][gslices_one_res_to_other_res]

inputs_data  = inputs['uz/uz/uz']
temp_uz['g'] = inputs_data[:, :, :][gslices_one_res_to_other_res]

inputs_data  = inputs['pressure/pressure/pressure']
temp_p['g']  = inputs_data[:, :, :][gslices_one_res_to_other_res]

if hydro_or_mhd == 'mhd':
    inputs_data   = inputs['Afx/Afx/Afx']
    temp_Ax['g']  = inputs_data[:, :, :][gslices_one_res_to_other_res]
 
    inputs_data   = inputs['Afy/Afy/Afy']
    temp_Ay['g']  = inputs_data[:, :, :][gslices_one_res_to_other_res]
 
    inputs_data   = inputs['Afz/Afz/Afz']
    temp_Az['g']  = inputs_data[:, :, :][gslices_one_res_to_other_res]
     
    inputs_data   = inputs['Psi/Psi/Psi']
    temp_Psi['g'] = inputs_data[:, :, :][gslices_one_res_to_other_res]

inputs.close()

temp_ux.set_scales(1.5)
temp_uy.set_scales(1.5)
temp_uz.set_scales(1.5)
temp_p.set_scales(1.5)

if hydro_or_mhd == 'mhd':
    temp_Ax.set_scales(1.5)
    temp_Ay.set_scales(1.5)
    temp_Az.set_scales(1.5)
    temp_Psi.set_scales(1.5)

logger.info("IC data loading ended.")
###############################################



############################
##Build problem, BCs     ###
############################
if hydro_or_mhd == 'hydro':
    problem_vars = ['ux', 'uy', 'uz', 'p', 'Ox', 'Oy']
else:
    problem_vars = ['ux', 'uy', 'uz', 'p', 'Ox', 'Oy', 'Ax', 'Ay', 'Az', 'Psi', 'Bx', 'By']

    
problem                           = de.IVP(domain, variables = problem_vars)
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['MA']          = MA
problem.parameters['MA2']         = MA**2.0 #Alfven Mach number squared
problem.parameters['Re']          = Reynolds
problem.parameters['Rm']          = MagReynolds
problem.parameters['Lx']          = Lx
problem.parameters['Ly']          = Ly
problem.parameters['Lz']          = Lz
problem.parameters['D_Krook']     = D_Krook
problem.parameters['a']           = a
problem.parameters['costh']       = np.cos(theta * np.pi/180)
problem.parameters['sinth']       = np.sin(theta * np.pi/180)

    
#problem.substitutions['u_dotgrad(thing_1)']    = "ux*dx(thing_1) +uy*dy(thing_1) +uz*dz(thing_1)"
problem.substitutions['u_mean']                = "integ(ux, 'x', 'y')/(Lx*Ly)"
problem.substitutions['u_ref']                 = "tanh(z/a)"


if forced == True:
    if force_nature == 'force2':
        problem.substitutions['Fx']   = "-1/Re*( -2/(a**2)*tanh(z/a)*1/((cosh(z/a))**2) ) + D_Krook*(u_ref - u_mean)"
else:
    problem.substitutions['Fx']   = "0.0"

    
if hydro_or_mhd == 'mhd':
    problem.substitutions["Bz"]           = "dx(Ay)-dy(Ax)"
    problem.substitutions["Oz"]           = "dx(uy)-dy(ux)" #O = curl of velocity.
    
    problem.substitutions["Jx"]           = "dy(Bz)-dz(By)"
    problem.substitutions["Jy"]           = "dz(Bx)-dx(Bz)"
    problem.substitutions["Jz"]           = "dx(By)-dy(Bx)"
    
    problem.substitutions["Kx"]           = "dy(Oz)-dz(Oy)" #K = curl of vorticity.
    problem.substitutions["Ky"]           = "dz(Ox)-dx(Oz)"
    problem.substitutions["Kz"]           = "dx(Oy)-dy(Ox)"
    
    problem.substitutions['O_cross_u_x']  = "Oy*uz - Oz*uy"
    problem.substitutions['O_cross_u_y']  = "Oz*ux - Ox*uz"
    problem.substitutions['O_cross_u_z']  = "Ox*uy - Oy*ux"

    problem.substitutions['J_cross_B_x']  = "Jy*Bz - Jz*By"
    problem.substitutions['J_cross_B_y']  = "Jz*Bx - Jx*Bz"
    problem.substitutions['J_cross_B_z']  = "Jx*By - Jy*Bx"

    problem.substitutions['u_cross_B_x']  = "uy*Bz - uz*By"
    problem.substitutions['u_cross_B_y']  = "uz*Bx - ux*Bz"
    problem.substitutions['u_cross_B_z']  = "ux*By - uy*Bx"

    problem.add_equation("dt(ux) + dx(p) + Kx/Re + Jz*sinth/MA2               = -O_cross_u_x + J_cross_B_x/MA2 + Fx")
    problem.add_equation("dt(uy) + dy(p) + Ky/Re - Jz*costh/MA2               = -O_cross_u_y + J_cross_B_y/MA2     ")
    problem.add_equation("dt(uz) + dz(p) + Kz/Re - (Jx*sinth - Jy*costh)/MA2  = -O_cross_u_z + J_cross_B_z/MA2     ")
    problem.add_equation("dx(ux) + dy(uy) + dz(uz)                            = 0")
    problem.add_equation("Ox - (dy(uz) - dz(uy))                              = 0")
    problem.add_equation("Oy - (dz(ux) - dx(uz))                              = 0")

    problem.add_equation("dt(Ax) + dx(Psi) + Jx/Rm + uz*sinth                 = u_cross_B_x")
    problem.add_equation("dt(Ay) + dy(Psi) + Jy/Rm - uz*costh                 = u_cross_B_y")
    problem.add_equation("dt(Az) + dz(Psi) + Jz/Rm - (ux*sinth-uy*costh)      = u_cross_B_z")
    problem.add_equation("dx(Ax) + dy(Ay) + dz(Az)                            = 0")
    problem.add_equation("Bx - (dy(Az) - dz(Ay))                              = 0")
    problem.add_equation("By - (dz(Ax) - dx(Az))                              = 0")
    
else:
    problem.substitutions["Oz"]           = "dx(uy)-dy(ux)" #O = curl of velocity.
    
    problem.substitutions["Kx"]           = "dy(Oz)-dz(Oy)" #K = curl of vorticity.
    problem.substitutions["Ky"]           = "dz(Ox)-dx(Oz)"
    problem.substitutions["Kz"]           = "dx(Oy)-dy(Ox)"
    
    problem.substitutions['O_cross_u_x']  = "Oy*uz - Oz*uy"
    problem.substitutions['O_cross_u_y']  = "Oz*ux - Ox*uz"
    problem.substitutions['O_cross_u_z']  = "Ox*uy - Oy*ux"

    #problem.substitutions['L_fourier(thing)'] = "d(thing, x=2) + d(thing, y=2)"
    
    problem.add_equation("dt(ux) + dx(p) + Kx/Re   = -O_cross_u_x + Fx")
    problem.add_equation("dt(uy) + dy(p) + Ky/Re   = -O_cross_u_y     ")
    problem.add_equation("dt(uz) + dz(p) + Kz/Re   = -O_cross_u_z     ")
    problem.add_equation("dx(ux) + dy(uy) + dz(uz) = 0")
    problem.add_equation("Ox - (dy(uz) - dz(uy))   = 0")
    problem.add_equation("Oy - (dz(ux) - dx(uz))   = 0")
    

problem.add_bc("left(ux)        = -1")
problem.add_bc("right(ux)       =  1")
problem.add_bc("left(uy)        =  0")
problem.add_bc("right(uy)       =  0")
problem.add_bc("left(uz)        =  0")
problem.add_bc("right(uz)       =  0", condition="(nx != 0) or  (ny != 0)")
problem.add_bc("right(p)        =  0", condition="(nx == 0) and (ny == 0)")

if hydro_or_mhd == 'mhd':
    
    problem.add_bc("left(Ax)       =  0")
    problem.add_bc("right(Ax)      =  0")
    problem.add_bc("left(Ay)       =  0")
    problem.add_bc("right(Ay)      =  0")
    problem.add_bc("left(Psi)      =  0")
    problem.add_bc("right(Psi)     =  0")
    #problem.add_bc("right(Psi)     =  0", condition="(nx != 0) or  (ny != 0)")
    #problem.add_bc("right(Az)      =  0", condition="(nx == 0) and (ny == 0)")
    
    '''
    problem.add_bc("left(Ax)       =  0")
    problem.add_bc("right(Ax)      =  0")
    problem.add_bc("left(Ay)       =  0")
    problem.add_bc("right(Ay)      =  0")
    problem.add_bc("left(Az)      =  0")
    #problem.add_bc("right(Psi)     =  0")
    problem.add_bc("right(Az)     =  0", condition="(nx != 0) or  (ny != 0)")
    problem.add_bc("right(Psi)      =  0", condition="(nx == 0) and (ny == 0)")
    '''
#Timestepping
#ts = de.timesteppers.SBDF2 #To-be-used only if there is a severe memory limitation, for e.g., while using 1024^3. 
ts = de.timesteppers.RK443
solver =  problem.build_solver(ts)
logger.info('Solver built')


#######################
##Initial conditions###
####################### 
#Solver state
ux   = solver.state['ux']
uy   = solver.state['uy']
uz   = solver.state['uz']
p    = solver.state['p']
Ox   = solver.state['Ox']
Oy   = solver.state['Oy']

if hydro_or_mhd == 'mhd':
    Ax   = solver.state['Ax']
    Ay   = solver.state['Ay']
    Az   = solver.state['Az']
    Bx   = solver.state['Bx']
    By   = solver.state['By']
    Psi  = solver.state['Psi']


#ICs
if checkpoint_read == False:

    if initcond == 'noise':
        kx = domain.elements(0)
        ky = domain.elements(1)

        logger.info('Initial conditions are being prepared.')

        ######################################################################################

        ux.set_scales(1.5)
        uy.set_scales(1.5)
        uz.set_scales(1.5)
        p.set_scales(1.5)

        
        if hydro_or_mhd == 'mhd':
            Ax.set_scales(1.5)
            Ay.set_scales(1.5)
            Az.set_scales(1.5)  
            Psi.set_scales(1.5)  
    
        #Computing additional pressure u^2/2 as I'm using: Omega x u + grad(u^2/2) = u.grad(u) identity.
        usqrd_op = temp_ux*temp_ux + temp_uy*temp_uy + temp_uz*temp_uz
        usqrd_ev = usqrd_op.evaluate()
        usqrd_ev.set_scales(1.5)

        ux['g'] = temp_ux['g']
        uy['g'] = temp_uy['g']
        uz['g'] = temp_uz['g']
        if restart_after_core_crash == False:
            p['g']  = temp_p['g'] + usqrd_ev['g']/2
        else:
            p['g']  = temp_p['g'] #because when I restart the simulation after a core crash or after some time-integration, then the pressure data saved in the snapshot by Dedalus is already the fluid pressure plus |u|^2/2. Only when I start the simulation at t=0, my pressure saved in the IC data is only the fluid pressure.
        ux.set_scales(1)
        uy.set_scales(1)
        uz.set_scales(1)
        p.set_scales(1)
        
        if hydro_or_mhd == 'mhd':
            Ax['g'] = temp_Ax['g']
            Ay['g'] = temp_Ay['g']
            Az['g'] = temp_Az['g']
            Psi['g']= temp_Psi['g']
            Ax.set_scales(1)
            Ay.set_scales(1)
            Az.set_scales(1)
            Psi.set_scales(1)
       
    Ox.set_scales(1.5)
    Oy.set_scales(1.5)

    Ox['g'] = temp_uz.differentiate(y=1)['g'] - temp_uy.differentiate(z=1)['g']
    Oy['g'] = temp_ux.differentiate(z=1)['g'] - temp_uz.differentiate(x=1)['g']
    Ox.set_scales(1)
    Oy.set_scales(1)
    
    if hydro_or_mhd == 'mhd':
        Bx.set_scales(1.5)
        By.set_scales(1.5)
        Bx['g'] = temp_Az.differentiate(y=1)['g'] - temp_Ay.differentiate(z=1)['g']
        By['g'] = temp_Ax.differentiate(z=1)['g'] - temp_Az.differentiate(x=1)['g']
        Bx.set_scales(1)
        By.set_scales(1)


    logger.info('Initial conditions have been prepared. Proceeding next!')

    dt0 = 0.0001*Lx/nx #This st0 has been used in cfl condn. max_dt = 100*dt0
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
##CFL conditions###
###################
solver.stop_sim_time  = sim_time 
solver.stop_wall_time = wall_time 
solver.stop_iteration = max_steps

cfl        = flow_tools.CFL(solver, initial_dt=dt, cadence = 100, safety = 0.1, max_change = 1.5, max_dt = 100*dt0, threshold = 0.05) 
#I've played with changing cadence from 1 to 200, and I see dt often not changing even after >500 iterations once the nonlinear phase is reached.

cfl.add_velocities(('ux', 'uy', 'uz'))

if hydro_or_mhd == 'mhd':
    cfl.add_velocities(('Bx/MA', 'By/MA', 'Bz/MA')) #cfl.add_velocities(('Bx','Bz')) with 4*pi*rho = 1 is the same as Bx/MA for CFL conditions.



#Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
if hydro_or_mhd == 'hydro':
    flow.add_property("1/2* (ux*ux + uy*uy + uz*uz) ",    name='energy_blow_up_check')
    flow.add_property("     (Ox*Ox + Oy*Oy + Oz*Oz)/Re ", name='dissipation_blow_up_check')
if hydro_or_mhd == 'mhd':
    flow.add_property("1/2* ((ux*ux + uy*uy + uz*uz) + (Bx*Bx + By*By + Bz*Bz)/(MA**2) ) ",       name='energy_blow_up_check')
    flow.add_property("     ((Ox*Ox + Oy*Oy + Oz*Oz)/Re + (Jx*Jx + Jy*Jy + Jz*Jz)/Rm/(MA**2) ) ", name='dissipation_blow_up_check')

###################
##Analysis tasks###
###################
analysis = solver.evaluator.add_file_handler('%s' %path_2, sim_dt = 1, mode = 'append') 
#analysis.add_system(solver.state, layout='g')
analysis.add_task('ux',  name='ux', layout='g')
analysis.add_task('uy',  name='uy', layout='g')
analysis.add_task('uz',  name='uz', layout='g')
analysis.add_task('p',   name='p',  layout='g')

if hydro_or_mhd == 'mhd':
    analysis.add_task('Ax',   name='Ax',  layout='g')
    analysis.add_task('Ay',   name='Ay',  layout='g')
    analysis.add_task('Az',   name='Az',  layout='g')
    analysis.add_task('Psi',  name='Psi', layout='g')
    #analysis.add_task('Bx',   name='Bx',  layout='g')
    #analysis.add_task('By',   name='By',  layout='g')
    #analysis.add_task('Jx',   name='Jx',  layout='g')
    #analysis.add_task('Jy',   name='Jy',  layout='g')
    #analysis.add_task('Ox',   name='Ox',  layout='g')
    #analysis.add_task('Oy',   name='Oy',  layout='g')

#This is only to continue the integration with restart.h5 file
final_chpt = solver.evaluator.add_file_handler('%s' %path_4, sim_dt = checkpoint_time, mode = 'overwrite') # sim_dt = sim_time-1 is because I have tried checking running the code and it turns  out that dedalus records data always t = actual_time-1 as the first frame is missed out somehow.
final_chpt.add_system(solver.state)



###################
##Actual Solver ###
###################
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #logger.info('Started loop')
        dt = cfl.compute_dt()
        #logger.info('dt computed: %0.10f' %dt)
        solver.step(dt)
        #logger.info('solver stepped')
        if solver.iteration % 100 == 0:
            logger.info('Iteration: %i, Time: %e, max_energy: %f, max_dissipation: %f, dt: %e' %(solver.iteration, solver.sim_time, flow.max('energy_blow_up_check'), flow.max('dissipation_blow_up_check'), dt))
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
    
    
    
    ##################################
    ##Saving the parameters choosen###
    ##################################
        
    if checkpoint_read == False:
        import json

        person_dict = {"Iterations": solver.iteration,
        "Sim end time": solver.sim_time,
        "Run time": (end_time - start_time), 
        "hydro_or_mhd": hydro_or_mhd,
        "theta": theta,
        "MA": MA,
        "Reynolds": Reynolds,          
        "MagReynolds": MagReynolds,
        "Schmidt": Schmidt,          
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz,          
        "nx": nx,
        "ny": ny,
        "nz": nz,          
        "sim_time": sim_time,
        "checkpoint_time": checkpoint_time,
        "wall_time": wall_time,          
        "forced": forced,
        "force_nature": force_nature,
        "passive_scalar": passive_scalar,  
        "initcond": initcond, 
        "checkpoint_read": checkpoint_read,
        "run_number": run_number,
        "month_of_run": month_of_run,          
        "script_version_number": script_version_number,
        "submit_version_number": submit_version_number,          
        "a": a,
        "D_Krook": D_Krook              
        }


        filename_param = '%s/KH_parameters.txt' %path_1
        json_file = open(filename_param, "w")
        json.dump(person_dict, json_file)
        json_file.close()
    
    logger.info('The simulation has ended! Merging files now!')
    
    
    
    
    
    ########################################
    ##Merging files from mpi parallel run###
    ########################################
    
    from dedalus.tools import post
    post.merge_process_files("%s" %path_2, cleanup=True)
    post.merge_process_files("%s" %path_4, cleanup=True)



logger.info('Final check! Yaay! You are amazing!')

