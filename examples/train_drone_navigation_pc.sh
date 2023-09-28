#!/bin/sh

#stage1
mpiexec -np 6 python examples/MPI_RL_drone_pc_train1.py && \
mpiexec -np 6 python examples/MPI_RL_drone_pc_evaluation1.py && \
mpiexec -np 6 python examples/MPI_RL_drone_pc_train2.py && \
mpiexec -np 6 python examples/MPI_RL_drone_pc_evaluation2.py