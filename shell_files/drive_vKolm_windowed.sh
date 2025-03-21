#!/bin/bash
#!
#! Example SLURM job script for Peta4-Skylake (Skylake CPUs, OPA)
#! Last updated: Mon 13 Nov 12:25:17 GMT 2017
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#SBATCH -p rrk26,rrk26-himem
#SBATCH --output=slurm_windowed_%A_%a.out
#SBATCH --nodes=1
#SBATCH -J window
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1

mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
np=$SLURM_TASKS_PER_NODE

#! ###########################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load miniforge3

#! Insert additional module load commands after this line if needed:
source activate dedalus

#! Full path to application executable:
application="python $HOME/projects/vKolmogorov3D/drive_vKolm_windowed.py"

#! Work directory (i.e. where the job will run):
workdir="$HOME/projects/vKolmogorov3D/"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                                # in which sbatch is run.
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets

CMD="srun --mpi=pmi2 -n ${SLURM_CPUS_PER_TASK} $application $SLURM_ARRAY_TASK_ID $*"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

#! if [ "$SLURM_JOB_NODELIST" ]; then
#!         #! Create a machine file:
#!         export NODEFILE=`generate_pbs_nodefile`
#!         cat $NODEFILE | uniq > machine.file.$JOBID
#!         echo -e "\nNodes allocated:\n================"
#!         #! echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
#! fi

echo -e "\nNodes allocated:\n================"

echo -e "\nnumtasks=$numtasks, numnodes=$SLURM_JOB_NUM_NODES, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
