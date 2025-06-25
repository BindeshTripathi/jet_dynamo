! Initial condition for the vector potential.
! This file contains the expression used for the initial 
! vector potential. You can use temporary real arrays R1-R3 
! of size (1:nx,1:ny,ksta:kend) and temporary complex arrays 
! C1-C8 of size (1:nz,1:ny,ista:iend) to do intermediate 
! computations. The variable a0 should control the global 
! amplitude of the initial condition, and variables 
! aparam0-9 can be used to control the amplitudes of 
! individual terms. At the end, the three components of the 
! potential in spectral space should be stored in the arrays 
! ax, ay, and az.

! Null vector potential

!!$omp parallel do if (iend-ista.ge.nth) private (j,k)
!      DO i = ista,iend
!!$omp parallel do if (iend-ista.lt.nth) private (k)
!         DO j = 1,ny
!            DO k = 1,nz
!               ax(k,j,i) = 0.0_GP
!               ay(k,j,i) = 0.0_GP
!               az(k,j,i) = 0.0_GP
!            END DO
!         END DO
!      END DO

! Superposition of ABC flows or some perturbations
!     kdn    : minimum wave number
!     kup    : maximum wave number
!     aparam0: A amplitude
!     aparam1: B amplitude
!     aparam2: C amplitude
!     aparam3: In terms of 2*pi, box length along z.
!     aparam4: In terms of 2*pi, box length along y.
!     aparam5: In terms of 2*pi, box length along x.
!     aparam6: Dimensional box length along z. That is, without 2*pi.
!     aparam7: the width of the gaussian envelope of the initial perturbations. (aparam7=2 can be chosen in the file "/bin/parameter.inp".)

!$omp parallel do if (kend-ksta.ge.nth) private (j,i)
      DO k = ksta,kend
!$omp parallel do if (kend-ksta.lt.nth) private (i)
         DO j = 1,ny
            DO i = 1,nx

            R1(i,j,k) = 0.
            R2(i,j,k) = 0.
            R3(i,j,k) = 0.

!Hermiticity demands summation over postive ki (positive n) only if m=0.
!The ICs are of the form: b = curl of a, where
!the vector potential a = amp/(m**2+n**2)*cos(m*x+n*y+phi_random)*(exp(-((z-z1)/aparam7)**2)+exp(-((z-z2)/aparam7)**2)).
!For every horizontal wavenumber (m,n), phi_random is a random phase, lying between (-pi,pi). 
!By hand, I am prescribing below random numbers from a random number generator.
               DO ki = 1,INT(kup)
                  R1(i,j,k) = R1(i,j,k) + aparam1*1/(0**2+ki**2)*COS(2*pi*0*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(50)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + aparam1*1/(0**2+ki**2)*COS(2*pi*0*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-42)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + aparam1*1/(0**2+ki**2)*COS(2*pi*0*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-81)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
               END DO

!When ki=0, hermicity is to be imposed, but the following equations satisfy it, because
!I have written separate DO loops for separate non-negative values of one of the wavenumbers: 0, 1, 2, 3, and 4. 
!Since cosine is used, no negative values (-1,-2,-3, and -4) are needed for that wavenumber, due to Hermiticity.
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + aparam1*1/(1**2+ki**2)*COS(2*pi*1*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(56)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + aparam1*1/(1**2+ki**2)*COS(2*pi*1*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(85)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + aparam1*1/(1**2+ki**2)*COS(2*pi*1*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-50)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
               END DO
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + aparam1*1/(2**2+ki**2)*COS(2*pi*2*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-26)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + aparam1*1/(2**2+ki**2)*COS(2*pi*2*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-1)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + aparam1*1/(2**2+ki**2)*COS(2*pi*2*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-24)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
               END DO
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + aparam1*1/(3**2+ki**2)*COS(2*pi*3*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(84)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + aparam1*1/(3**2+ki**2)*COS(2*pi*3*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(18)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + aparam1*1/(3**2+ki**2)*COS(2*pi*3*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-21)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
               END DO
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + aparam1*1/(4**2+ki**2)*COS(2*pi*4*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(25)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + aparam1*1/(4**2+ki**2)*COS(2*pi*4*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(42)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + aparam1*1/(4**2+ki**2)*COS(2*pi*4*(real(i,kind=GP)-1)/real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(20)/100*3.1415926535897932) * &
                              (EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*aparam6)/aparam7)**2) + &
                              EXP(-((aparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*aparam6)/aparam7)**2))
               END DO

            END DO
         END DO
      END DO

!Now, let us compute curl of the vector R, by first transforming it to spectral space and then computing curl in spectral space. 
      CALL fftp3d_real_to_complex(planrc,R1,C1,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planrc,R2,C2,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planrc,R3,C3,MPI_COMM_WORLD)

      CALL rotor3(C2,C3,ax,1)
      CALL rotor3(C1,C3,ay,2)
      CALL rotor3(C1,C2,az,3)
