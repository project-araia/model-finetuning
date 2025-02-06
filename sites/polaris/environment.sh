# Load MPI module. This should be available as standard module on a cluster.
# If not, build your own MPI and update PATH, LD_LIBRARY_PATH
module load PrgEnv-nvhpc/8.5.0

# Set MPI_HOME by quering path loaded by site module
export MPI_HOME=$(which mpicc | sed s/'\/bin\/mpicc'//)

module load cray-python/3.11.5

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

export MPICH_CC=gcc-12
export MPICH_CXX=g++-12
export MPICH_FC=gfortran-12
export MPICH_F90=gfortran-12
export PATH=$HOME/.local/bin:$PATH
