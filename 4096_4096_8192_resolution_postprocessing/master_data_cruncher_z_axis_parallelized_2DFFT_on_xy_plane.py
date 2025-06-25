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
count_t        = add_this_to_count_t_for_jobarray_handling + int(sys.argv[3])  #So, actual count_t is add_this_to_count_t_for_jobarray_handling + count_t. Job array does not accept values larger than 999.
nx             = int(sys.argv[4])
ny             = int(sys.argv[5])
nz             = int(sys.argv[6])
add_this_to_z_axis_ind_for_jobarray_handling = int(sys.argv[7])
z_axis_ind     = add_this_to_z_axis_ind_for_jobarray_handling + int(sys.argv[8]) #So, actual z_axis_ind is add_this_to_z_axis_ind_for_jobarray_handling + z_axis_ind. Job array does not accept values larger than 999.
#parallelization along the z-axis, by using 2D FFTs on the x-y plane. I am using cylindrical shells here.

z_derivative_accuracy = "fourth_order" #Options are "second_order" and "fourth_order". 

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

# #################################################################
# kx_array = np.zeros((int(nx/2+1), ny, nz))
# ky_array = np.zeros((int(nx/2+1), ny, nz))
# kz_array = np.zeros((int(nx/2+1), ny, nz))

# current_time = datetime.now()
# print("Step 0 done. Current time:", current_time.strftime("%H:%M:%S"))

# for jj in range(0,ny):
#     for kk in range(0,nz):
#         kx_array[:int(nx/2+1),jj,kk] = np.linspace(0, 2*np.pi/Lx*nx/2, int(nx/2+1))

# for ii in range(0,int(nx/2+1)):
#     for kk in range(0,nz):
#         ky_array[ii,0:int(ny/2+1),kk] =        np.linspace(0, 2*np.pi/Ly*int(ny/2), int(ny/2+1))
#         ky_array[ii,int(ny/2+1): ,kk] = (-1) * np.linspace(0, 2*np.pi/Ly*int(ny/2), int(ny/2+1))[::-1][1:-1]

# for ii in range(0,int(nx/2+1)):
#     for jj in range(0,ny):
#         kz_array[ii,jj,0:int(nz/2+1)] =         np.linspace(0, 2*np.pi/Lz*int(nz/2), int(nz/2+1))
#         kz_array[ii,jj,int(nz/2+1): ] =  (-1) * np.linspace(0, 2*np.pi/Lz*int(nz/2), int(nz/2+1))[::-1][1:-1]


#################################################################
kx_array_2D = np.zeros((int(nx/2+1), ny))
ky_array_2D = np.zeros((int(nx/2+1), ny))
kz_array_2D = np.zeros((int(nx/2+1), nz))

current_time = datetime.now()
print("Step 0 done. Current time:", current_time.strftime("%H:%M:%S"))

for jj in range(0,ny):
    kx_array_2D[:int(nx/2+1),jj] = np.linspace(0, 2*np.pi/Lx*nx/2, int(nx/2+1))

for ii in range(0,int(nx/2+1)):
    ky_array_2D[ii,0:int(ny/2+1)] =        np.linspace(0, 2*np.pi/Ly*int(ny/2), int(ny/2+1))
    ky_array_2D[ii,int(ny/2+1): ] = (-1) * np.linspace(0, 2*np.pi/Ly*int(ny/2), int(ny/2+1))[::-1][1:-1]

for ii in range(0,int(nx/2+1)):
    kz_array_2D[ii,0:int(nz/2+1)] =        np.linspace(0, 2*np.pi/Lz*int(nz/2), int(nz/2+1))
    kz_array_2D[ii,int(nz/2+1): ] = (-1) * np.linspace(0, 2*np.pi/Lz*int(nz/2), int(nz/2+1))[::-1][1:-1]

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
    print("pyfftw is not installed, so scipy's serial fft will be used")
pyfftw_import = False

# #################################################################
# def FFT_rc(data_in):
#     return np.fft.rfftn(data_in.T,  axes=(0,1,2), norm='forward').T
# #################################################################
# def IFFT_cr(data_in):
#     return np.fft.irfftn(data_in.T, axes=(0,1,2), norm="forward").T
# #################################################################
# def FFT_cc_1D(data_in):
#     return np.fft.fft(data_in.T, norm='forward').T
# #################################################################
# def IFFT_cc_1D(data_in):
#     return np.fft.ifft(data_in.T, norm="forward").T
# #################################################################
# def FFT_rc_1D(data_in):
#     return np.fft.rfft(data_in.T, norm='forward').T
# #################################################################
# def IFFT_cr_1D(data_in):
#     return np.fft.irfft(data_in.T, norm="forward").T
#################################################################
def FFT_rc_2D(data_in):
    return np.fft.rfftn(data_in.T,  axes=(0,1), norm='forward').T
#################################################################
def IFFT_cr_2D(data_in):
    return np.fft.irfftn(data_in.T, axes=(0,1), norm="forward").T
#################################################################
# def dx(data_in):
#     return IFFT_cr(FFT_rc(data_in)*1.0j*kx_array)
# #################################################################
# def dy(data_in):
#     return IFFT_cr(FFT_rc(data_in)*1.0j*ky_array)
# #################################################################
# def dz(data_in):
#     return IFFT_cr(FFT_rc(data_in)*1.0j*kz_array)
# #################################################################
# def dz_1D(data_in): #1D derivative along z. This is useful to compute derivative for a given kx, ky.
#     return IFFT_cc_1D(FFT_cc_1D(data_in)*1.0j*kz_array[0,0,:])
#################################################################
def dx_2D_rc_and_cr(data_in): #derivative along x with 2D (x,y) FFT. This is useful to compute derivative for a given x and y for 4k x 4k x 8k resolution dataset.
    return IFFT_cr_2D(FFT_rc_2D(data_in)*1.0j*kx_array_2D)
#################################################################
def dy_2D_rc_and_cr(data_in): #derivative along y with 2D (x,y) FFT. This is useful to compute derivative for a given x and y for 4k x 4k x 8k resolution dataset.
    return IFFT_cr_2D(FFT_rc_2D(data_in)*1.0j*ky_array_2D)
# #################################################################
# def dz_1D_rc_and_cr(data_in): #1D derivative along z for a given x and y location. This is useful to compute derivative for a given x and y for 4k x 4k x 8k resolution dataset.
#     return IFFT_cr_1D(FFT_rc_1D(data_in)*1.0j*kz_array_1D)
# #################################################################
# def dz_squared_1D_rc_and_cr(data_in): #1D derivative along z for a given x and y location. This is useful to compute derivative for a given x and y for 4k x 4k x 8k resolution dataset.
#     return IFFT_cr_1D(FFT_rc_1D(data_in)*(-1)*kz_array_1D**2)
# #################################################################
# def dz_2D_rc_and_cr(data_in): #1D derivative along z for a given x and y location. This is useful to compute derivative for a given x and y for 4k x 4k x 8k resolution dataset.
#     return IFFT_cr_2D(FFT_rc_2D(data_in)*1.0j*kz_array_2D)
# #################################################################
# def dz_squared_2D_rc_and_cr(data_in): #1D derivative along z for a given x and y location. This is useful to compute derivative for a given x and y for 4k x 4k x 8k resolution dataset.
#     return IFFT_cr_2D(FFT_rc_2D(data_in)*(-1)*kz_array_2D**2)
# #################################################################
# def curl_op(dat_x, dat_y, dat_z, axis):
#     if axis=="x":
#         return (dy(dat_z) - dz(dat_y))
#     elif axis=="y":
#         return (dz(dat_x) - dx(dat_z))
#     elif axis=="z":
#         return (dx(dat_y) - dy(dat_x))
# #################################################################
# def kz_to_z_for_a_given_kxky(data_in):
#     return IFFT_cc_1D(data_in.T)
# #################################################################
# def z_to_kz_for_a_given_kxky(data_in):
#     return FFT_cc_1D(data_in.T)
#################################################################
def wavenumber_filter(filter_type, data_in, k_inside, k_outside):
    data_temp = FFT_rc_2D(data_in)

    if filter_type=="cylindrical_shells":
        for ii in range(0,int(nx/2+1)):
            for jj in range(0,ny):
                k_mag = np.sqrt(kx_array_2D[ii,0]**2+ky_array_2D[0,jj]**2)
                if (k_mag>=k_inside) and (k_mag<k_outside):
                    None
                else:
                    data_temp[ii,jj] *= 0

    return IFFT_cr_2D(data_temp)
#################################################################
def transfer_function(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, type_of_C_field): #A_i (B_j \cdot \nabla C_i) is the form of the nonlinearity considered here. 
    if type_of_C_field=="u_type":

        filelist = sorted(glob.glob(path+'vx.*.out'))
        dz_Cx = dz_of_a_or_u_on_xyplane(filelist, count_t, z_axis_ind)

        filelist = sorted(glob.glob(path+'vy.*.out'))
        dz_Cy = dz_of_a_or_u_on_xyplane(filelist, count_t, z_axis_ind)

    elif type_of_C_field=="b_type":

        ax_filelist = sorted(glob.glob(path+'ax.*.out'))
        ay_filelist = sorted(glob.glob(path+'ay.*.out'))
        az_filelist = sorted(glob.glob(path+'az.*.out'))

        dz_Cx = dy_2D_rc_and_cr(- dx_2D_rc_and_cr(xy_slice_data_reader("a", "x", count_t, z_axis_ind)) - dy_2D_rc_and_cr(xy_slice_data_reader("a", "y", count_t, z_axis_ind)) ) - dz_squared_of_a_or_u_on_xyplane(ay_filelist, count_t, z_axis_ind) #dz(bx) = dy dz(az) - dz^2(ay) = dy (-dx(ax)-dy(ay)) - dz^2 ay.

        dz_Cy = dx_2D_rc_and_cr(  dx_2D_rc_and_cr(xy_slice_data_reader("a", "x", count_t, z_axis_ind)) + dy_2D_rc_and_cr(xy_slice_data_reader("a", "y", count_t, z_axis_ind)) ) + dz_squared_of_a_or_u_on_xyplane(ax_filelist, count_t, z_axis_ind) #dz(by) = -dx dz(az) + dz^2(ax) = dx (dx(ax)+dy(ay)) + dz^2 ay.


    return np.mean( Ax*( Bx*dx_2D_rc_and_cr(Cx) + By*dy_2D_rc_and_cr(Cx) + Bz*dz_Cx ) + Ay*( Bx*dx_2D_rc_and_cr(Cy) + By*dy_2D_rc_and_cr(Cy) + Bz*dz_Cy ) + Az*( Bx*dx_2D_rc_and_cr(Cz) + By*dy_2D_rc_and_cr(Cz) + Bz*(-dx_2D_rc_and_cr(Cx) - dy_2D_rc_and_cr(Cy)) ) ) #Here, in the last term, I have used dz_Cz = -dx(Cx) - dy(Cy) ) = -dx_2D_rc_and_cr(Cx) - dy_2D_rc_and_cr(Cy).
#################################################################
def xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind):
    with open(filelist[count_t], 'rb') as f:
        f.seek(8*nx*ny*z_axis_ind) #8 for 8 bytes of a 64-bit floating point number
        return np.frombuffer(f.read(8*nx*ny), dtype=dtype).reshape((nx,ny),order='F')
# #################################################################
# def extraction_of_z_lines_at_a_given_x_and_y_location_from_binary_file(filelist, count_t, x_axis_ind, y_axis_ind):
#     temp_arr = np.zeros(nz, dtype=dtype)
#     with open(filelist[count_t], 'rb') as f:
#         for z_axis_ind in range(0, nz):
#             f.seek(8*nx*ny*z_axis_ind + 8*nx*y_axis_ind + 8*x_axis_ind) #8 for 8 bytes of a 64-bit floating point number
#             temp_arr[z_axis_ind] = np.frombuffer(f.read(8), dtype=dtype).reshape(1,order='F')
#     return temp_arr
# #################################################################
# def dz_of_a_or_u_on_xyplane(filelist, count_t, z_axis_ind):
#     temp_arr_xy = np.zeros((nx,ny), dtype=dtype)
#     for x_axis_ind in range(0,nx):
#         for y_axis_ind in range(0,ny):
#             temp_arr_xy[x_axis_ind, y_axis_ind] = dz_1D_rc_and_cr(extraction_of_z_lines_at_a_given_x_and_y_location_from_binary_file(filelist, count_t, x_axis_ind, y_axis_ind))[z_axis_ind]
#         print("hi")
#     return temp_arr_xy
#################################################################
def dz_of_a_or_u_on_xyplane(filelist, count_t, z_axis_ind):
    if z_derivative_accuracy=="second_order":
        if z_axis_ind==0:
            return 1/(2*Lz/nz)*( xy_slice_extraction_from_binary_file(filelist, count_t, 1) - xy_slice_extraction_from_binary_file(filelist, count_t, (nz-1)%nz) )
        else:
            return 1/(2*Lz/nz)*( xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-1)%nz) )
    elif z_derivative_accuracy=="fourth_order":
        if z_axis_ind==0:
            return 1/(12*Lz/nz)*( -xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+2)%nz) + 8*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - 8*xy_slice_extraction_from_binary_file(filelist, count_t, (nz-1)%nz) + xy_slice_extraction_from_binary_file(filelist, count_t, (nz-2)%nz) )
        elif z_axis_ind==1:
            return 1/(12*Lz/nz)*( -xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+2)%nz) + 8*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - 8*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-1)%nz) + xy_slice_extraction_from_binary_file(filelist, count_t, (nz-1)%nz) )
        else:
            return 1/(12*Lz/nz)*( -xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+2)%nz) + 8*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - 8*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-1)%nz) + xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-2)%nz) )

#################################################################
def dz_squared_of_a_or_u_on_xyplane(filelist, count_t, z_axis_ind):
    if z_derivative_accuracy=="second_order":
        if z_axis_ind==0:
            return 1/((Lz/nz)**2)*( xy_slice_extraction_from_binary_file(filelist, count_t, 1) + xy_slice_extraction_from_binary_file(filelist, count_t, nz-1) -  2* xy_slice_extraction_from_binary_file(filelist, count_t, 0)  )
        else:
            return 1/((Lz/nz)**2)*( xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) + xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-1)%nz) - 2* xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind) )
    elif z_derivative_accuracy=="fourth_order":
        if z_axis_ind==0:
            return 1/(12*(Lz/nz)**2)*( -xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+2)%nz) + 16*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - 30*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind)%nz) + 16*xy_slice_extraction_from_binary_file(filelist, count_t, (nz-1)%nz) - xy_slice_extraction_from_binary_file(filelist, count_t, (nz-2)%nz) )
        elif z_axis_ind==1:
            return 1/(12*(Lz/nz)**2)*( -xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+2)%nz) + 16*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - 30*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind)%nz) + 16*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-1)%nz) - xy_slice_extraction_from_binary_file(filelist, count_t, (nz-1)%nz) )
        else:
            return 1/(12*(Lz/nz)**2)*( -xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+2)%nz) + 16*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind+1)%nz) - 30*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind)%nz) + 16*xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-1)%nz) - xy_slice_extraction_from_binary_file(filelist, count_t, (z_axis_ind-2)%nz) )
    
#################################################################  
def xy_slice_data_reader(field_type, component, count_t, z_axis_ind):
    if field_type=='u':
        if component=="x":
            filelist = sorted(glob.glob(path+'vx.*.out'))
            return xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind)
        if component=="y":
            filelist = sorted(glob.glob(path+'vy.*.out'))
            return xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind)
        if component=="z":
            filelist = sorted(glob.glob(path+'vz.*.out'))
            return xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind)

    if field_type=='a':
        if component=="x":
            filelist = sorted(glob.glob(path+'ax.*.out'))
            return xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind)
        if component=="y":
            filelist = sorted(glob.glob(path+'ay.*.out'))
            return xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind)
        if component=="z":
            filelist = sorted(glob.glob(path+'az.*.out'))
            return xy_slice_extraction_from_binary_file(filelist, count_t, z_axis_ind)

    if field_type=='b':
        ax_filelist = sorted(glob.glob(path+'ax.*.out'))
        ay_filelist = sorted(glob.glob(path+'ay.*.out'))
        az_filelist = sorted(glob.glob(path+'az.*.out'))

        if component=="x":
            return dy_2D_rc_and_cr(xy_slice_extraction_from_binary_file(az_filelist, count_t, z_axis_ind)) - dz_of_a_or_u_on_xyplane(ay_filelist, count_t, z_axis_ind)
        if component=="y":
            return  dz_of_a_or_u_on_xyplane(ax_filelist, count_t, z_axis_ind) - dx_2D_rc_and_cr(xy_slice_extraction_from_binary_file(az_filelist, count_t, z_axis_ind))
        if component=="z":
            return dx_2D_rc_and_cr(xy_slice_extraction_from_binary_file(ay_filelist, count_t, z_axis_ind)) - dy_2D_rc_and_cr(xy_slice_extraction_from_binary_file(ax_filelist, count_t, z_axis_ind))

# #################################################################
# def data_reader(field_type, component, count_t):
#     if field_type=='u':
#         vx_filelist = sorted(glob.glob(path+'vx.*.out'))
#         vy_filelist = sorted(glob.glob(path+'vy.*.out'))
#         vz_filelist = sorted(glob.glob(path+'vz.*.out'))

#         if component=="x":
#             return np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F')
#         if component=="y":
#             return np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F')
#         if component=="z":
#             return np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F')

#     if field_type=='O':
#         vx_filelist = sorted(glob.glob(path+'vx.*.out'))
#         vy_filelist = sorted(glob.glob(path+'vy.*.out'))
#         vz_filelist = sorted(glob.glob(path+'vz.*.out'))

#         if component=="x":
#             return curl_op(np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "x")
#         if component=="y":
#             return curl_op(np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "y")
#         if component=="z":
#             return curl_op(np.fromfile(vx_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(vy_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(vz_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "z")


#     if field_type=='a':
#         ax_filelist = sorted(glob.glob(path+'ax.*.out'))
#         ay_filelist = sorted(glob.glob(path+'ay.*.out'))
#         az_filelist = sorted(glob.glob(path+'az.*.out'))

#         if component=="x":
#             return np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F')
#         if component=="y":
#             return np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F')
#         if component=="z":
#             return np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F')

#     if field_type=='b':
#         ax_filelist = sorted(glob.glob(path+'ax.*.out'))
#         ay_filelist = sorted(glob.glob(path+'ay.*.out'))
#         az_filelist = sorted(glob.glob(path+'az.*.out'))

#         if component=="x":
#             return curl_op(np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "x")
#         if component=="y":
#             return curl_op(np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "y")
#         if component=="z":
#             return curl_op(np.fromfile(ax_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(ay_filelist[count_t],dtype=dtype).reshape(shape,order='F'),
#                         np.fromfile(az_filelist[count_t],dtype=dtype).reshape(shape,order='F'), "z")

#################################################################

current_time = datetime.now()
print("Step 1 done. Current time:", current_time.strftime("%H:%M:%S"))

nfiles = np.size(sorted(glob.glob(path+'vz.*.out')))

#ux_data = data_reader('u', 'x', count_t)
#uy_data = data_reader('u', 'y', count_t)
#uz_data = data_reader('u', 'z', count_t)

#Ox_data = data_reader('O', 'x', count_t)
#Oy_data = data_reader('O', 'y', count_t)
#Oz_data = data_reader('O', 'z', count_t)

#ax_data = data_reader('a', 'x', count_t)
#ay_data = data_reader('a', 'y', count_t)
#az_data = data_reader('a', 'z', count_t)

#bx_data = data_reader('b', 'x', count_t)
#by_data = data_reader('b', 'y', count_t)
#bz_data = data_reader('b', 'z', count_t)

print(count_t)

Pi_u_to_u = np.zeros(k_shell_arr.shape[0], dtype=np.float64)
Pi_b_to_b = np.zeros(k_shell_arr.shape[0], dtype=np.float64)

for pp in range(0,k_shell_arr.shape[0]-1):
    k_inside = 0 #This is ONLY for energy flux, where wavenumbers are cumulatively added. This is NOT for energy transfer where individual wavenumber is selected, not cumulative.
    k_outside = k_shell_arr[pp] #This is for run567 where I use logarithmic binning of wavenumbers using cylindrical shells

    #k_inside = k_shell_arr[pp]
    #k_outside = k_shell_arr[pp+1]

    ux_P = wavenumber_filter(filter_type, xy_slice_data_reader('u', 'x', count_t, z_axis_ind), k_inside, k_outside)
    uy_P = wavenumber_filter(filter_type, xy_slice_data_reader('u', 'y', count_t, z_axis_ind), k_inside, k_outside)
    uz_P = wavenumber_filter(filter_type, xy_slice_data_reader('u', 'z', count_t, z_axis_ind), k_inside, k_outside)
    bx_P = wavenumber_filter(filter_type, xy_slice_data_reader('b', 'x', count_t, z_axis_ind), k_inside, k_outside)
    by_P = wavenumber_filter(filter_type, xy_slice_data_reader('b', 'y', count_t, z_axis_ind), k_inside, k_outside)
    bz_P = wavenumber_filter(filter_type, xy_slice_data_reader('b', 'z', count_t, z_axis_ind), k_inside, k_outside)

    Pi_u_to_u[pp] = (-1) * transfer_function(ux_P, uy_P, uz_P, xy_slice_data_reader('u', 'x', count_t, z_axis_ind), xy_slice_data_reader('u', 'y', count_t, z_axis_ind), xy_slice_data_reader('u', 'z', count_t, z_axis_ind), xy_slice_data_reader('u', 'x', count_t, z_axis_ind), xy_slice_data_reader('u', 'y', count_t, z_axis_ind), xy_slice_data_reader('u', 'z', count_t, z_axis_ind), "u_type")

    Pi_b_to_b[pp] = (-1) * transfer_function(bx_P, by_P, bz_P, xy_slice_data_reader('u', 'x', count_t, z_axis_ind), xy_slice_data_reader('u', 'y', count_t, z_axis_ind), xy_slice_data_reader('u', 'z', count_t, z_axis_ind), xy_slice_data_reader('b', 'x', count_t, z_axis_ind), xy_slice_data_reader('b', 'y', count_t, z_axis_ind), xy_slice_data_reader('b', 'z', count_t, z_axis_ind), "b_type")

    print(f"Time: {count_t}, pp: {pp}, z_axis_ind: {z_axis_ind}, To-be-z-avgd Pi_u_to_u: {Pi_u_to_u[pp]}, To-be-z-avgd Pi_b_to_b: {Pi_b_to_b[pp]}")
    current_time = datetime.now()
    print("Current time:", current_time.strftime("%H:%M:%S"))

################################
hf = h5py.File(f'/anvil/projects/x-phy130027/phd2019/processed_data_for_dynamo_paper/dynamo_paper_1_publication_figure_data/{directory_name_for_saving_output}/{run_number_IVP}_time_{count_t}_all_Log_Radial_Wavenumbers_for_z_axis_ind_{z_axis_ind}.h5', 'w')

g11 = hf.create_group('Pi_u_to_u')
g11.create_dataset('Pi_u_to_u',data=Pi_u_to_u)

g12 = hf.create_group('Pi_b_to_b')
g12.create_dataset('Pi_b_to_b',data=Pi_b_to_b)

hf.close()

print(f"Job completed for count_t: {count_t}, at z_axis_ind: {z_axis_ind}.")
