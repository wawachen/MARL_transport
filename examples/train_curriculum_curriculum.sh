#!/bin/sh

# # 2 to 4
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

python examples/MPI_RL_drone_pc_train1_final_shared_curriculum.py  \
--scenario-name="navigation_curriculum_curriculum" \
--n-agents=3 \
--max-episodes=1000 \
--max-episode-len=50 \
--lr-actor=1e-4 \
--lr-critic=1e-3 \
--batch-size=1024 \
--evaluate-episodes=20 \
--evaluate-episode-len=30 \
--load-type="three" \
--stage=1 \
--field-size=10 

# mpiexec -np 1 python examples/MPI_RL_drone_pc_train1_final_shared_wt.py  \
# --scenario-name="navigation_curriculum_wt_adaptive_1" \
# --n-agents=4 \
# --max-episodes=1500 \
# --max-episode-len=60 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="four" \
# --stage=2 \
# --field-size=10 

