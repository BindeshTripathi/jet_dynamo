#!/bin/bash
######SLURM submit file for Paraview server filename: paraview_server_job.submit
#SBATCH -J DEDALUS        # Job Name
#SBATCH -o DEDALUS.out%j    # Output and error file name (%j expands to jobID)
#SBATCH -e DEDALUS.err%j
#SBATCH --ntasks 128
#SBATCH -N 1            # Total # of nodes (now required)
#SBATCH -p wholenode #wide #wholenode  # Queue (partition) name -- normal, development, skx-normal, etc.
#SBATCH -t 24:00:00     # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=btripathi@wisc.edu 
#SBATCH --mail-type=all   #begin, end, fail, or all
#####SBATCH -A TG-PHY130027
#SBATCH -A phy130027
#######Set a port for running the paraview server connection
export PV_ACCESSPORT=11111

module load gcc/11.2.0
module load openmpi/4.0.6 
module load paraview/5.9.1 

#######For single CPU server job
pvserver  --server-port=$PV_ACCESSPORT 

######For parallel server
######mpirun -np 128 $SLURM_NTASKS pvserver --server-port=$PV_ACCESSPORT
