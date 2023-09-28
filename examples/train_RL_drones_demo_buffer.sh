
# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_demowp_buffer.py \
# --scenario-name="navigation_mpi_demowp_buffer" \
# --n-agents=3 \
# --env-size=1 \
# --max-episodes=1500 \
# --max-episode-len=50 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=150 \
# --evaluate-episode-len=50 \
# --load-type="three" \
# --evaluate


mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_demowp_buffer.py \
--scenario-name="navigation_mpi_demowp_buffer" \
--n-agents=4 \
--env-size=1 \
--max-episodes=2000 \
--max-episode-len=60 \
--lr-actor=1e-4 \
--lr-critic=1e-3 \
--batch-size=1024 \
--field-size=10 \
--local-sight=2.0 \
--evaluate-episodes=200 \
--evaluate-episode-len=60 \
--load-type="four" \
--evaluate


# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_demowp.py \
# --scenario-name="navigation_mpi_demowp" \
# --n-agents=4 \
# --env-size=1 \
# --max-episodes=2000 \
# --max-episode-len=60 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=200 \
# --evaluate-episode-len=60 \
# --load-type="four" \
# --evaluate


# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_demowp.py \
# --scenario-name="navigation_mpi_demowp" \
# --n-agents=6 \
# --env-size=1 \
# --max-episodes=5000 \
# --max-episode-len=70 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=400 \
# --evaluate-episode-len=70 \
# --load-type="six" \
# --evaluate

# evaluation
# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_demowp.py \
# --scenario-name="analysis/drone3/demo/1" \
# --n-agents=3 \
# --env-size=1 \
# --max-episodes=2000 \
# --max-episode-len=70 \
# --lr-actor=1e-4 \
# --lr-critic=1e-3 \
# --batch-size=1024 \
# --field-size=10 \
# --local-sight=2.0 \
# --evaluate-episodes=20 \
# --evaluate-episode-len=50 \
# --load-type="three" \
# --evaluate




