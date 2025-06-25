! User-defined forcing scheme.
! A forcing scheme can be implemented in this file, which
! is used by the code when 'rand=3' is set in 'parameter.dat'.
! This scheme is executed every 'fstep' time steps. See the 
! folder 'examples' for an example. If not needed, this file
! can be left empty.
! Forcing arrays are complex (in Fourier space) of size
! (n,n,ista:iend) and are called:
!       (fx,fy,fz)   for the velocity 
!       (mx,my,mz)   for the e.m.f. (magnetic field)
!       (fs,fs1,...) for scalar fields
!       (fre,fim)    for quantum solvers
! You can use temporary real arrays R1-R3 of size
! (1:n,1:n,ksta:kend) and temporary complex arrays C1-C8 of
! size (n,n,ista:iend) to do intermediate computations,
! and two real arrays Faux1 and Faux2 of size (10) to store
! information of the history of the forcing if needed.

!     vparam6: Dimensional box length along z. That is, without 2*pi.

!!!$omp parallel do if (iend-ista.ge.nth) private (j,k)
!      DO i = ista,iend
!!!$omp parallel do if (iend-ista.lt.nth) private (k)
!         DO j = 1,ny
!            DO k = 1,nz
!               fy(k,j,i) = 0.0_GP
!               fz(k,j,i) = 0.0_GP
!            END DO
!         END DO
!      END DO


!$omp parallel do if (kend-ksta.ge.nth) private (j,i)
      DO k = ksta,kend
!$omp parallel do if (kend-ksta.lt.nth) private (i)
         DO j = 1,ny
            DO i = 1,nx

               R1(i,j,k) = 0 * (TANH((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)) &
                           -TANH((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)) &
                           -1) / 20.0
!Note that in the expression for R1, I have introduced a factor of 1/2,
!which is 1/D_Krook, and D_Krook is 2 here. If D_Krook is to be changed
!here, it should be changed in the RK time stepper routine as well, which is
!available in mhd_rkstep2.f90. Tripathi. March 18, 2025.

            END DO
         END DO
      END DO

      CALL fftp3d_real_to_complex(planrc,R1,fx,MPI_COMM_WORLD)
!      CALL normalize(fx,fy,fz,f0,1,MPI_COMM_WORLD)
