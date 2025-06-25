import numpy as np
import h5py
import time
import shutil, os
import sys
import glob as glob
import string as string
from mpi4py import MPI
CW = MPI.COMM_WORLD
from datetime import datetime

#################################
#Check the parameters below

run_number_IVP = sys.argv[1]
add_this_to_count_t_for_jobarray_handling = int(sys.argv[2]) #e.g., this can be 0, or 1000, or 2000, etc. 
count_t        = int(sys.argv[3]) + add_this_to_count_t_for_jobarray_handling  #So, actual count_t is add_this_to_count_t_for_jobarray_handling + count_t. Job array does not accept values larger than 999.
nx             = int(sys.argv[4])
ny             = int(sys.argv[5])
nz             = int(sys.argv[6])
pp             = int(sys.argv[7])

# Box size
Lx = 10*np.pi
Ly = 10*np.pi
Lz = 20*np.pi

#Shell-decomposed energy transfer function.
k_logarithmic_index_max = int(4*np.log(nx/8)/np.log(2) + 1)
k_logarithmic_indices = np.append(1, 2**(2+1/4*(-1 + np.linspace(0, k_logarithmic_index_max, k_logarithmic_index_max+1))))
k_shell_arr = 2*np.pi/Lx * k_logarithmic_indices

filter_type = "cylindrical_shells"

t_step_for_analysis = 1

path = '/anvil/scratch/x-btripathi/oct2020/%s/' %run_number_IVP

################################
shape = (nx,ny,nz)
dtype = np.float64

directory_name_for_saving_output = "cross_scale_energy_flux_%s_%s" %(filter_type, run_number_IVP)

########################################
start_time = time.time()

file_dir_1 = '/anvil/projects/x-phy130027/phd2019/processed_data_for_dynamo_paper/dynamo_paper_1_publication_figure_data/%s' %(directory_name_for_saving_output)
path_1 = file_dir_1

if CW.rank == 0:
    access_rights = 0o755

    try:
        os.mkdir(path_1, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_1)

#################################################################
kx_array = np.zeros((int(nx/2+1), ny, nz))
ky_array = np.zeros((int(nx/2+1), ny, nz))
kz_array = np.zeros((int(nx/2+1), ny, nz))

current_time = datetime.now()
print("Step 0 done. Current time:", current_time.strftime("%H:%M:%S"))

for jj in range(0,ny):
    for kk in range(0,nz):
        kx_array[:int(nx/2+1),jj,kk] = np.linspace(0, 2*np.pi/Lx*nx/2, int(nx/2+1))
        
for ii in range(0,int(nx/2+1)):
    for kk in range(0,nz):
        ky_array[ii,0:int(ny/2+1),kk] =        np.linspace(0, 2*np.pi/Ly*int(ny/2), int(ny/2+1))
        ky_array[ii,int(ny/2+1): ,kk] = (-1) * np.linspace(0, 2*np.pi/Ly*int(ny/2), int(ny/2+1))[::-1][1:-1]
        
for ii in range(0,int(nx/2+1)):
    for jj in range(0,ny):
        kz_array[ii,jj,0:int(nz/2+1)] =         np.linspace(0, 2*np.pi/Lz*int(nz/2), int(nz/2+1))
        kz_array[ii,jj,int(nz/2+1): ] =  (-1) * np.linspace(0, 2*np.pi/Lz*int(nz/2), int(nz/2+1))[::-1][1:-1]
        
#################################################################

import numpy.fft as fft
import numpy as np
import multiprocessing
try: 
    import pyfftw
    pyfftw_import = True
    pyfftw.interfaces.cache.enable()
    threads = multiprocessing.cpu_count()
except ImportError:
    print("pyfftw is not installed, now using scipy's serial fft")
pyfftw_import = False

#################################################################
def FFT_rc(data_in):
    return np.fft.rfftn(data_in.T,  axes=(0,1,2), norm='forward').T
#################################################################
def IFFT_cr(data_in):
    return np.fft.irfftn(data_in.T, axes=(0,1,2), norm="forward").T
#################################################################
def FFT_cc_1D(data_in):
    return np.fft.fft(data_in.T, norm='forward').T
#################################################################
def IFFT_cc_1D(data_in):
    return np.fft.ifft(data_in.T, norm="forward").T
#################################################################
def dx(data_in):
    return IFFT_cr(FFT_rc(data_in)*1.0j*kx_array)
#################################################################
def dy(data_in):
    return IFFT_cr(FFT_rc(data_in)*1.0j*ky_array)
#################################################################
def dz(data_in):
    return IFFT_cr(FFT_rc(data_in)*1.0j*kz_array)
#################################################################
def dz_1D(data_in): #1D derivative along z. This is useful to compute derivative for a given kx, ky.
    return IFFT_cc_1D(FFT_cc_1D(data_in)*1.0j*kz_array[0,0,:])
#################################################################
def curl_op(dat_x, dat_y, dat_z, axis):
    if axis=="x":
        return (dy(dat_z) - dz(dat_y))
    elif axis=="y":
        return (dz(dat_x) - dx(dat_z))
    elif axis=="z":
        return (dx(dat_y) - dy(dat_x))
#################################################################
def kz_to_z_for_a_given_kxky(data_in):
    return IFFT_cc_1D(data_in.T)
#################################################################
def z_to_kz_for_a_given_kxky(data_in):
    return FFT_cc_1D(data_in.T)
#################################################################
def wavenumber_filter(filter_type, data_in, k_inside, k_outside):
    data_temp = FFT_rc(data_in)
#    data_temp_2 = 0*data_temp
    
    if filter_type=="cylindrical_shells":
        for ii in range(0,int(nx/2+1)):
            for jj in range(0,ny):
                k_mag = np.sqrt(kx_array[ii,0,0]**2+ky_array[0,jj,0]**2)
                if (k_mag>=k_inside) and (k_mag<k_outside):
                    None
                else:
                    data_temp[ii,jj,:] *= 0
                    
    if filter_type=="spherical_shells":
        for ii in range(0,int(nx/2+1)):
            for jj in range(0,ny):
                for kk in range(0,nz):
                    k_mag = np.sqrt(kx_array[ii,0,0]**2+ky_array[0,jj,0]**2+kz_array[0,0,kk]**2)
                    if (k_mag>=k_inside) and (k_mag<k_outside):
                        None
                    else:
                        data_temp[ii,jj,kk] *= 0
                        
    return IFFT_cr(data_temp)
#################################################################
def transfer_function(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz): #A_i (B_j \cdot \nabla C_i) is the form of the nonlinearity considered here.
    return np.mean( Ax*( Bx*dx(Cx) + By*dy(Cx) + Bz*dz(Cx) ) + Ay*( Bx*dx(Cy) + By*dy(Cy) + Bz*dz(Cy) ) + Az*( Bx*dx(Cz) + By*dy(Cz) + Bz*dz(Cz) ) )
#################################################################
def data_reader(field_type, component, count_t):
    if field_type=='u':
        vx_filelist = sorted(glob.glob(path+'vx.*.out'))
        vy_filelist = sorted(glob.glob(path+'vy.*.out'))
        vz_filelist = sorted(glob.glob(path+'vz.*.out'))

        if component=="x":
            return np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F')
        if component=="y":
            return np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F')
        if component=="z":
            return np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F')
    
    if field_type=='O':
        vx_filelist = sorted(glob.glob(path+'vx.*.out'))
        vy_filelist = sorted(glob.glob(path+'vy.*.out'))
        vz_filelist = sorted(glob.glob(path+'vz.*.out'))

        if component=="x":
            return curl_op(np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "x")
        if component=="y":
            return curl_op(np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "y")
        if component=="z":
            return curl_op(np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "z")


    if field_type=='a':
        ax_filelist = sorted(glob.glob(path+'ax.*.out'))
        ay_filelist = sorted(glob.glob(path+'ay.*.out'))
        az_filelist = sorted(glob.glob(path+'az.*.out'))

        if component=="x":
            return np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F')
        if component=="y":
            return np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F')
        if component=="z":
            return np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F')
    
    if field_type=='b':
        ax_filelist = sorted(glob.glob(path+'ax.*.out'))
        ay_filelist = sorted(glob.glob(path+'ay.*.out'))
        az_filelist = sorted(glob.glob(path+'az.*.out'))

        if component=="x":
            return curl_op(np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "x")
        if component=="y":
            return curl_op(np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "y")
        if component=="z":
            return curl_op(np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
                        np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "z")

#################################################################

current_time = datetime.now()
print("Step 1 done. Current time:", current_time.strftime("%H:%M:%S"))

nfiles = np.size(sorted(glob.glob(path+'vz.*.out')))

print(count_t)
current_time = datetime.now()
print("Step 2 done. Current time:", current_time.strftime("%H:%M:%S"))

#for pp in range(0,k_shell_arr.shape[0]-1):
k_inside = 0 #This is ONLY for energy flux, where wavenumbers are cumulatively added.
k_outside = k_shell_arr[pp] 

#k_inside = k_shell_arr[pp]
#k_outside = k_shell_arr[pp+1]

ux_P = wavenumber_filter(filter_type, data_reader('u', 'x', count_t), k_inside, k_outside)

current_time = datetime.now()
print("Step 3 done. Current time:", current_time.strftime("%H:%M:%S"))

uy_P = wavenumber_filter(filter_type, data_reader('u', 'y', count_t), k_inside, k_outside)
uz_P = wavenumber_filter(filter_type, data_reader('u', 'z', count_t), k_inside, k_outside)
bx_P = wavenumber_filter(filter_type, data_reader('b', 'x', count_t), k_inside, k_outside)
by_P = wavenumber_filter(filter_type, data_reader('b', 'y', count_t), k_inside, k_outside)
bz_P = wavenumber_filter(filter_type, data_reader('b', 'z', count_t), k_inside, k_outside)

current_time = datetime.now()
print("Step 4 done. Current time:", current_time.strftime("%H:%M:%S"))

Pi_u_to_u = (-1) * transfer_function(ux_P, uy_P, uz_P, data_reader('u', 'x', count_t), data_reader('u', 'y', count_t), data_reader('u', 'z', count_t), data_reader('u', 'x', count_t), data_reader('u', 'y', count_t), data_reader('u', 'z', count_t))

print(f"Time: {count_t}, pp: {pp}, Pi_u_to_u: {Pi_u_to_u}")

Pi_b_to_b = (-1) * transfer_function(bx_P, by_P, bz_P, data_reader('u', 'x', count_t), data_reader('u', 'y', count_t), data_reader('u', 'z', count_t), data_reader('b', 'x', count_t), data_reader('b', 'y', count_t), data_reader('b', 'z', count_t))

print(f"Time: {count_t}, pp: {pp}, Pi_u_to_u: {Pi_u_to_u}, Pi_b_to_b: {Pi_b_to_b}")
current_time = datetime.now()
print("Current time:", current_time.strftime("%H:%M:%S"))

print(f"Job complete for {count_t}.")

################################
hf = h5py.File(f'/anvil/projects/x-phy130027/phd2019/processed_data_for_dynamo_paper/dynamo_paper_1_publication_figure_data/{directory_name_for_saving_output}/{run_number_IVP}_time_{count_t}_RadialWavenumberIndex_{pp}.h5', 'w')

g11 = hf.create_group('Pi_u_to_u')
g11.create_dataset('Pi_u_to_u',data=Pi_u_to_u)

g12 = hf.create_group('Pi_b_to_b')
g12.create_dataset('Pi_b_to_b',data=Pi_b_to_b)

hf.close()
