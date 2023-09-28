
python examples/MPI_RL_drone_navigation_V0_attention.py \
--scenario-name="navigation_mpi_V0_attention_aw_4m" \
--n-agents=3 \
--env-size=1 \
--max-episodes=1500 \
--max-episode-len=50 \
--lr-actor=1e-4 \
--lr-critic=1e-3 \
--batch-size=1024 \
--field-size=10 \
--local-sight=4.0 \
--evaluate-episodes=20 \
--evaluate-episode-len=30 \
--load-type="three" \
--is-local-obs


# python examples/MPI_RL_drone_navigation_V0_attention.py \
# --scenario-name="navigation_mpi_V0_attention" \
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

# python examples/MPI_RL_drone_navigation_V0_attention.py \
# --scenario-name="navigation_mpi_V0_attention" \
# --n-agents=6 \
# --env-size=1 \
# --max-episodes=3500 \
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
# mpiexec -np 1 python examples/MPI_RL_drone_navigation_V0_attention.py \
# --scenario-name="analysis/drone3/attention/1" \
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



