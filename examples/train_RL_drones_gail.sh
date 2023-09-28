
python examples/main_GAIL.py \
--scenario-name="navigation_mpi_gail" \
--n-agents=3 \
--env-size=1 \
--max-episodes=500 \
--max-episode-len=50 \
--field-size=10 \
--evaluate-episodes=20 \
--evaluate-episode-len=30 \
--load-type="three" 

# python examples/MPI_RL_drone_navigation_V0.py \
# --scenario-name="navigation_mpi_V0" \
# --n-agents=4 \
# --env-size=1 \
# --max-episodes=2000 \
# --max-episode-len=60 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="four"

# python examples/MPI_RL_drone_navigation_V0.py \
# --scenario-name="navigation_mpi_V0" \
# --n-agents=6 \
# --env-size=1 \
# --max-episodes=5000 \
# --max-episode-len=70 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=30 \
# --load-type="six"

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




