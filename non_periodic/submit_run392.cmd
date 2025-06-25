#!/bin/bash
#SBATCH -J DEDALUS        # Job Name
#SBATCH -o DEDALUS.out%j    # Output and error file name (%j expands to jobID)
#SBATCH -e DEDALUS.err%j
#####SBATCH -n 128           # Total number of mpi tasks requested
##SBATCH --ntasks-per-node 128  #64
#SBATCH -N 256 # Total # of nodes (now required)
#SBATCH -p wide #wide #wholenode  # Queue (partition) name -- normal, development, skx-normal, etc.
#SBATCH -t 48:00:00     # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=btripathi@wisc.edu 
#SBATCH --mail-type=all   #begin, end, fail, or all
#####SBATCH -A TG-PHY130027
#SBATCH -A phy130027
###activatededalus
######ibrun ./gene_stampede           # Run the MPI executable named a.out
mpiexec -n 32768 python3 MHD_KH_run392.py    # Make sure you run the same verison of script that you have on MHD_KH_v8.py
######python3 dye_plot_v1.py
