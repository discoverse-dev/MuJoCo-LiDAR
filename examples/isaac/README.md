# IsaacLab integration

## Installation

Install Isaac Sim and IsaacLab following: https://github.com/isaac-sim/IsaacLab

```bash
uv pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

```bash
# install python module (for rsl-rl)
./isaaclab.sh -i rsl_rl
# run script for rl training of the teacher agent
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless
# run script for distilling the teacher agent into a student agent
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
# run script for playing the student with 64 environments
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point
```