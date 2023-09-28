
python examples/main_MAPPO.py \
--scenario-name="navigation_mpi_MAPPO" \
--n-agents=3 \
--env-size=1 \
--max-episodes=1500 \
--episode_limit=50 \
--field-size=10 \
--local-sight=2.0 \
--evaluate-episodes=20 \
--evaluate-episode-len=30 \
--load-type="three" \
--evaluate

# python examples/main_MAPPO.py \
# --scenario-name="navigation_mpi_MAPPO" \
# --n-agents=4 \
# --env-size=1 \
# --max-episodes=2000 \
# --episode_limit=60 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="four" \
# --evaluate

# python examples/main_MAPPO.py \
# --scenario-name="navigation_mpi_MAPPO" \
# --n-agents=6 \
# --env-size=1 \
# --max-episodes=4000 \
# --episode_limit=70 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="six" \
# --evaluate

#evaluation
# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0.py \
# --scenario-name="analysis/drone3/scratch/1" \
# --n-agents=3 \
# --env-size=1 \
# --max-episodes=1500 \
# --max-episode-len=50 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=50 \
# --load-type="three" \
# --evaluate




