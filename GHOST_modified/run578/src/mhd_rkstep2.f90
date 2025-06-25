! Step 2 of Runge-Kutta for the MHD equations
! Computes the nonlinear terms and evolves the equations in dt/o

         CALL rotor3(ay,az,C7,1)
         CALL rotor3(ax,az,C8,2)
         CALL rotor3(ax,ay,C9,3)
         CALL prodre3(vx,vy,vz,C10,C11,C12)
         CALL prodre3(C7,C8,C9,C13,C14,C15)
         IF ((trans.eq.1).and.(times.eq.0).and.(bench.eq.0).and.(o.eq.ord)) &
            CALL entrans(C1,C2,C3,C13,C14,C15,ext,2)
         CALL nonlin3(C10,C11,C12,C13,C14,C15,C16,1)
         CALL nonlin3(C10,C11,C12,C13,C14,C15,C17,2)
         CALL nonlin3(C10,C11,C12,C13,C14,C15,C10,3)
         IF ((trans.eq.1).and.(times.eq.0).and.(bench.eq.0).and.(o.eq.ord)) &
            CALL entrans(C7,C8,C9,C16,C17,C10,ext,4)
         CALL vector3(vx,vy,vz,C7,C8,C9,C11,C12,C13)
         CALL gauge3(C11,C12,C13,C7,1)
         CALL gauge3(C11,C12,C13,C8,2)
         CALL gauge3(C11,C12,C13,C9,3)
!Tripathi introduced the next line "C14=vx", before vx is modified to
!represent Laplacian of vx. C14 is not used below anywhere, except now
!that I use in the forcing function, wher C14 represents instantaneous
!mean flow—instantaneous value is needed in 2nd order higher order RK
!method, where this file is executed 2 or more number of times. 
!March 19, 2025.
!         C14(k,j,i) = vx(k,j,i)

         CALL laplak3(vx,vx)
         CALL laplak3(vy,vy)
         CALL laplak3(vz,vz)
         CALL laplak3(ax,ax)
         CALL laplak3(ay,ay)
         CALL laplak3(az,az)
         IF ((trans.eq.1).and.(times.eq.0).and.(bench.eq.0).and.(o.eq.ord)) &
            THEN
            CALL entrans(C1,C2,C3,C16,C17,C10,ext,1)
            CALL entrans(C4,C5,C6,C7,C8,C9,ext,0)
            CALL rotor3(C8,C9,C11,1)
            CALL rotor3(C7,C9,C12,2)
            CALL rotor3(C7,C8,C13,3)
            CALL entrans(C1,C2,C3,C11,C12,C13,ext,3)
         ENDIF

         rmp = 1./real(o,kind=GP)
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
         DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
         DO j = 1,ny
         DO k = 1,nz
            IF ((kn2(k,j,i).le.kmax).and.(kn2(k,j,i).ge.tiny)) THEN
!The following IF clause is introduced by Tripathi. Mar 18, 2025. Note
!the use of -C14(k,j,i)/2. Here, C14 is velocity in the Fourier space. The
!factor 2 is the D_Krook (forcing) timescale. The remainder of the
!foring, which is independent of the instantaneous mean flow, is
!incorporated using the initialfv.f90 forcing routine, available in src.
!               IF ((i.eq.1).and.(j.eq.1)) THEN
!                  vx(k,j,i) = C1(k,j,i)+dt*(nu*vx(k,j,i)-0*nu*vx(k,j,i)+C16(k,j,i) &
!                  +0*fx(k,j,i)-0*vx(k,1,1)/(-(k/10)**2)*1/2-0*C14(k,1,1)/2)*rmp      
!               ELSE 
!                  vx(k,j,i) = C1(k,j,i)+dt*(nu*vx(k,j,i)+C16(k,j,i) &
!                  +0*fx(k,j,i))*rmp
!               ENDIF
!               IF ((kx(i)==0).and.(ky(j)==0)) THEN
!                  vx(k,j,i) = C1(k,j,i)+dt*(nu*vx(k,j,i)-0*nu*vx(k,j,i)+C16(k,j,i) &
!                              +0*fx(k,j,i)-0*vx(k,1,1)/(-(k/10)**2)*1/2-0*C14(k,1,1)/2)*rmp
!               vx(k,j,i) = C1(k,j,i)+dt*((nu+1.0/2.0*10.0/k*10.0/k)*vx(k,j,i)+C16(k,j,i) &
!              +fx(k,j,i))*rmp
!               IF (kz(k)==0) THEN
!                  vx(k,j,i) = C1(k,j,i)+dt*(nu*vx(k,j,i)+C16(k,j,i) &
!                 +fx(k,j,i))*rmp
!               ELSE
               
! Tripathi notes. March 26, 2025.
! Here, the 1st stage of RK222 is u(t0+dt/2) = (u(t0) + dt/2*RHS(t=t0))     * 1 / (1+nu*ksquared*dt/2). 
! The 2nd stage of RK222 is       u(t0+dt)   = (u(t0) + dt * RHS(t=t0+dt/2))* 1 / (1+nu*ksquared*dt  ).
! Where RHS represents all nonlinear and forcing terms except the visco-resistive disipation.

               IF (((kx(i)==0).and.(ky(j)==0)).and.(kz(k).ne.0)) THEN
                  vx(k,j,i) = C1(k,j,i)
!                  vx(k,j,i) = C1(k,j,i)+dt*rmp*(C16(k,j,i) &
!              +fx(k,j,i)+(1.0/20.0*1.0/kz(k)*1.0/kz(k))*vx(k,j,i))
! In the above line, I am solving the eqn: dt(ux)
! = nonlinear terms + mean flow forcing part 1 + mean flow forcing part 2 + (mean flow forcing part 3 + nu*nabla^2*ux) . Here,  mean flow forcings are:
! mean flow forcing part 1 (fx is ux_ref(z)/DKrook) + 
! mean flow forcing part 2 (- ux_instantaneous/DKrook) +
! mean flow forcing part 3 (mean flow forcing part 3 + nu*nabla^2*ux = 0). This means mean flow forcing part 3 = -nu*nabla^2*ux, 
! which is used to obtain a mean-flow equilibrium, to which we add perturbations and do linear analysis.
               ELSE
                  vx(k,j,i) = (C1(k,j,i)+dt*rmp*EXP(nu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp*(2.0-1.0/rmp))*(C16(k,j,i) &
              +fx(k,j,i)))*EXP(-nu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp)
               ENDIF
               vy(k,j,i) = (C2(k,j,i)+dt*rmp*EXP(nu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp*(2.0-1.0/rmp))*(C17(k,j,i) &
              +fy(k,j,i)))*EXP(-nu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp)
               vz(k,j,i) = (C3(k,j,i)+dt*rmp*EXP(nu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp*(2.0-1.0/rmp))*(C10(k,j,i) &
              +fz(k,j,i)))*EXP(-nu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp)
               ax(k,j,i) = (C4(k,j,i)+dt*rmp*EXP(mu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp*(2.0-1.0/rmp))*(C7(k,j,i)  &
              +mx(k,j,i)))*EXP(-mu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp)
               ay(k,j,i) = (C5(k,j,i)+dt*rmp*EXP(mu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp*(2.0-1.0/rmp))*(C8(k,j,i)  &
              +my(k,j,i)))*EXP(-mu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp)
               az(k,j,i) = (C6(k,j,i)+dt*rmp*EXP(mu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp*(2.0-1.0/rmp))*(C9(k,j,i)  &
              +mz(k,j,i)))*EXP(-mu*(kx(i)*kx(i)+ky(j)*ky(j)+kz(k)*kz(k))*dt*rmp)
            ELSE
               vx(k,j,i) = 0.0_GP
               vy(k,j,i) = 0.0_GP
               vz(k,j,i) = 0.0_GP
               ax(k,j,i) = 0.0_GP
               ay(k,j,i) = 0.0_GP
               az(k,j,i) = 0.0_GP
            ENDIF
         END DO
         END DO
         END DO
