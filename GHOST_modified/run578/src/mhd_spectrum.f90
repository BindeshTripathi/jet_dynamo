! Spectra computed in MHD runs

            CALL spectrum(vx,vy,vz,ext,1,1)
            CALL spectrum(ax,ay,az,ext,0,1)
            CALL crosspec(vx,vy,vz,ax,ay,az,ext)
