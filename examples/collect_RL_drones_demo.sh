#!/bin/sh

python examples/orca_drone_navigation_demo_v0.py \
--scenario-name="orca_navigation_drone_gail" \
--max-episodes=3000 \
--max-episode-len=25 \
--n-agents=3 \
--evaluate-episode-len=30 \
--evaluate-episodes=20 \
--field-size=10 \
--load-type="three"


# python examples/orca_drone_navigation_demo_v0.py \
# --scenario-name="orca_navigation_drone" \
# --max-episodes=100 \
# --max-episode-len=25 \
# --n-agents=4 \
# --evaluate-episode-len=30 \
# --evaluate-episodes=20 \
# --field-size=10 \
# --load-type="four"


# python examples/orca_drone_navigation_demo_v0.py \
# --scenario-name="orca_navigation_drone" \
# --max-episodes=35 \
# --max-episode-len=60 \
# --n-agents=6 \
# --evaluate-episode-len=30 \
# --evaluate-episodes=20 \
# --field-size=10 \
# --load-type="six"

# python examples/orca_drone_navigation_demo_v0.py \
# --scenario-name="orca_navigation_drone" \
# --max-episodes=20 \
# --max-episode-len=70 \
# --n-agents=6 \
# --evaluate-episode-len=30 \
# --evaluate-episodes=20 \
# --field-size=10 \
# --load-type="six"

# python examples/orca_drone_navigation_demo_v0.py \
# --scenario-name="Evaluate_orca_navigation_drone" \
# --max-episodes=1000 \
# --max-episode-len=25 \
# --n-agents=6 \
# --evaluate-episode-len=35 \
# --evaluate-episodes=1 \
# --field-size=10 \
# --load-type="six" \
# --evaluate

