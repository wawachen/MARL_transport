#!/bin/sh

#2 to 4
# mpiexec -np 1 python examples/MPI_RL_drone_pc_train1_final_shared.py  \
# --scenario-name="navigation_curriculum" \
# --n-agents=2 \
# --max-episodes=1000 \
# --max-episode-len=50 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="four" \
# --stage=1 \
# --field-size=5

mpiexec -np 1 python3 examples/MPI_RL_drone_pc_train1_final_shared.py  \
--scenario-name="navigation_curriculum" \
--n-agents=6 \
--max-episodes=1000 \
--max-episode-len=60 \
--lr-actor=1e-4 \
--lr-critic=1e-3 \
--batch-size=1024 \
--evaluate-episodes=20 \
--evaluate-episode-len=30 \
--load-type="six" \
--stage=2 \
--field-size=5 \
--evaluate 
