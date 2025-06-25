! Initial condition for the velocity.
! This file contains the expression used for the initial 
! velocity field. You can use temporary real arrays R1-R3 
! of size (1:nx,1:ny,ksta:kend) and temporary complex arrays 
! C1-C8 of size (nz,ny,ista:iend) to do intermediate 
! computations. The variable u0 should control the global 
! amplitude of the velocity, and variables vparam0-9 can be
! used to control the amplitudes of individual terms. At the
! end, the three components of the velocity in spectral
! space should be stored in the arrays vx, vy, and vz.

! Superposition of ABC flows or some perturbations
!     kdn    : minimum wave number
!     kup    : maximum wave number
!     vparam0: A amplitude
!     vparam1: B amplitude
!     vparam2: C amplitude
!     vparam3: In terms of 2*pi, box length along z.
!     vparam4: In terms of 2*pi, box length along y.
!     vparam5: In terms of 2*pi, box length along x.
!     vparam6: Dimensional box length along z. That is, without 2*pi.
!     vparam7: the width of the gaussian envelope of the initial perturbations. (vparam7=2 can be chosen in the file "/bin/parameter.inp".)


!$omp parallel do if (kend-ksta.ge.nth) private (j,i)
      DO k = ksta,kend
!$omp parallel do if (kend-ksta.lt.nth) private (i)
         DO j = 1,ny
            DO i = 1,nx

            R1(i,j,k) = 0.
            R2(i,j,k) = 0.
            R3(i,j,k) = 0.

!Hermiticity demands summation over postive ki (positive n) only if m=0.
!The ICs are of the form: v = amp/sqrt(m**2+n**2)*cos(m*x+n*y+phi_random)*(exp(-((z-z1)/vparam7)**2)+exp(-((z-z2)/vparam7)**2)).
!For every horizontal wavenumber (m,n), phi_random is a random phase, lying between (-pi,pi). 
!By hand, I am prescribing below random numbers from a random number generator.
               DO ki = 1,INT(kup)
                  R1(i,j,k) = R1(i,j,k) + vparam1*1/SQRT(real(0,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*0*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-67)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + vparam1*1/SQRT(real(0,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*0*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(7)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + vparam1*1/SQRT(real(0,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*0*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-71)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
               END DO

!When ki=0, hermicity is to be imposed, but the following equations satisfy it, because
!I have written separate DO loops for separate non-negative values of one of the wavenumbers: 0, 1, 2, 3, and 4. 
!Since cosine is used, no negative values (-1,-2,-3, and -4) are needed for that wavenumber, due to Hermiticity.
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + vparam1*1/SQRT(real(1,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*1*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(61)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + vparam1*1/SQRT(real(1,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*1*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-44)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + vparam1*1/SQRT(real(1,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*1*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(0)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
               END DO
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + vparam1*1/SQRT(real(2,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*2*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(37)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + vparam1*1/SQRT(real(2,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*2*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(18)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + vparam1*1/SQRT(real(2,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*2*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(31)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
               END DO
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + vparam1*1/SQRT(real(3,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*3*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(6)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + vparam1*1/SQRT(real(3,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*3*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-79)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + vparam1*1/SQRT(real(3,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*3*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-20)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
               END DO
               DO ki = INT(kdn),INT(kup)
                  R1(i,j,k) = R1(i,j,k) + vparam1*1/SQRT(real(4,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*4*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(49)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R2(i,j,k) = R2(i,j,k) + vparam1*1/SQRT(real(4,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*4*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(-80)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
                  R3(i,j,k) = R3(i,j,k) + vparam1*1/SQRT(real(4,kind=GP)**2+real(ki,kind=GP)**2)*COS(2*pi*4*(real(i,kind=GP)-1)/ &
                              real(nx,kind=GP) + &
                              2*pi*ki*(real(j,kind=GP)-1)/real(ny,kind=GP) + ki/kup*(84)/100*3.1415926535897932) * &
                              (EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)/vparam7)**2) + &
                              EXP(-((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)/vparam7)**2))
               END DO

            END DO
         END DO
      END DO

!Now, let us compute curl of the vector R, by first transforming it to spectral space and then computing curl in spectral space. 
      CALL fftp3d_real_to_complex(planrc,R1,C1,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planrc,R2,C2,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planrc,R3,C3,MPI_COMM_WORLD)

      CALL rotor3(C2,C3,C4,1)
      CALL rotor3(C1,C3,vy,2)
      CALL rotor3(C1,C2,vz,3)

!Now, let us add the x-directed mean flow to the perturbations and save  such total vx in spectral space.

!$omp parallel do if (kend-ksta.ge.nth) private (j,i)
      DO k = ksta,kend
!$omp parallel do if (kend-ksta.lt.nth) private (i)
         DO j = 1,ny
            DO i = 1,nx

            R1(i,j,k) = 0.
            R2(i,j,k) = 0.
            R3(i,j,k) = 0.

            R1(i,j,k) = R1(i,j,k) &
                        +TANH((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.25*vparam6)) &
                        -TANH((vparam6*(real(k,kind=GP)-1)/real(nz,kind=GP) - 0.75*vparam6)) &
                        - 1
            END DO
         END DO
      END DO
    
      CALL fftp3d_real_to_complex(planrc,R1,C7,MPI_COMM_WORLD)
      
!Adding perturbations to the mean flow and saving such total vx in spectral space.
      DO i = ista,iend
          DO j = 1,ny
             DO k = 1,nz
                vx(k,j,i) = C4(k,j,i) + C7(k,j,i)
             END DO
          END DO
      END DO
