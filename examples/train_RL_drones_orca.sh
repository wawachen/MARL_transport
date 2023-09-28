
# python examples/MPI_RL_drone_navigation_V0_orca.py \
# --scenario-name="navigation_mpi_orca_stage200" \
# --n-agents=3 \
# --env-size=1 \
# --max-episodes=1500 \
# --max-episode-len=50 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=256 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="three" \
# --stage-t=400 #400

# python examples/MPI_RL_drone_navigation_V0_orca.py \
# --scenario-name="navigation_mpi_orca_stage200" \
# --n-agents=4 \
# --env-size=1 \
# --max-episodes=2000 \
# --max-episode-len=60 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=256 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="four" \
# --stage-t=200   #700

python examples/MPI_RL_drone_navigation_V0_orca.py \
--scenario-name="navigation_mpi_orca_stage" \
--n-agents=6 \
--env-size=1 \
--max-episodes=3500 \
--max-episode-len=70 \
--lr-actor=1e-4 \
--lr-critic=1e-3 \
--batch-size=512 \
--field-size=10 \
--local-sight=2.0 \
--evaluate-episodes=20 \
--evaluate-episode-len=70 \
--load-type="six" \
--stage-t=3500 \
--evaluate

#evaluation
# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_orca.py \
# --scenario-name="analysis/drone6/bc/1" \
# --n-agents=6 \
# --env-size=1 \
# --max-episodes=1500 \
# --max-episode-len=50 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=256 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=50 \
# --load-type="six" \
# --stage-t=200 \
# --evaluate
