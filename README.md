# ENPM 701 Grand Challenge — Final Project

## Overview

Autonomous robot that locates coloured blocks in an arena, picks them up, and delivers
them to a scoring zone.  The codebase is structured as a Python package (`fp/`) with a
live diagnostic REPL available during runs via `--repl`.

## Quick start

```bash
# From final_project/
python main.py                        # full autonomous run (waits for Enter)
python main.py --no-wait              # skip the Enter handshake
python main.py --repl                 # autonomous run with live REPL attached
python main.py --color red            # lock to red blocks only
python -m fp run --repl               # equivalent explicit form
```

## Package layout

```
final_project/
├── main.py                 # thin shim → fp.cli (preserves old invocation paths)
└── fp/
    ├── __init__.py
    ├── __main__.py         # python -m fp entry point
    ├── cli.py              # argparse (all subcommands) + main()
    │
    ├── hal.py              # GPIO, motors, encoders, gripper, ultrasonic, IMU
    ├── arena.py            # arena dimensions, walls, zones, QR_LANDMARKS
    ├── motion.py           # PID primitives, move_forward, pivot_to_heading, navigate_to
    ├── localization.py     # odometry thread, wall correction, sweep+fit, calibrate_initial_pose
    ├── navigation.py       # compatibility shim re-exporting motion + localization
    ├── ekf.py              # extended Kalman filter (predict / update_heading / update_wall / update_qr)
    ├── vision.py           # camera init, HSV block detection, estimate_block_pose, within_reach
    ├── qr_detection.py     # QR code decode + PnP pose
    ├── planner.py          # occupancy grid, A* path planning, frontier exploration
    ├── tuning.py           # runtime-tunable scalar parameters (get_param / set_param / list_params)
    ├── state_machine.py    # compatibility shim → fp.fsm
    ├── teleop.py           # interactive teleop modes (hal, navigation, odometry, …)
    │
    ├── fsm/
    │   ├── context.py      # RobotContext dataclass (pose, score, planner, camera, …)
    │   ├── runner.py       # event loop: state.step → next_state; FsmControl for REPL hooks
    │   └── states.py       # InitState, DiscoverState, ApproachState, GrabState,
    │                       #   DeliverState, AvoidState, UpdateState, CompleteState
    │
    └── repl/
        ├── repl.py         # REPL input loop (FSM on worker thread; REPL on main thread)
        ├── commands.py     # command registry and handlers
        └── tuning_diff.py  # session diff, runtime.toml persistence
```

## REPL commands

Start a run with `--repl`, then type commands at the `fp>` prompt:

| Command | Description |
|---|---|
| `pose` | Print current robot pose (x, y, heading_deg) |
| `cov` | Print EKF covariance diagonal |
| `state` | Print current FSM state name |
| `blocks` | Print blocks delivered, score, and target colour |
| `pause` | Request FSM pause (takes effect before the next state tick) |
| `resume` | Resume a paused FSM |
| `step` | Execute one FSM tick while paused, then re-pause |
| `force STATE` | Force FSM to a named state — e.g. `force DISCOVER` |
| `tune` | List all tuning params (`*` = changed this session) |
| `set KEY VALUE` | Mutate a tuning param live — e.g. `set ARRIVAL_TOLERANCE_CM 8.0` |
| `revert KEY` | Revert a param to its session-start value |
| `:diff` | Show all tuning changes made this session |
| `:save` | Persist session changes to `runtime.toml` (merged with existing) |
| `:reset` | Revert all params to session-start values |
| `help` | List available commands |
| `quit` / `exit` | Exit the REPL (FSM keeps running if not paused) |

## Subcommands (test/teleop modes)

```
python main.py hal                         # interactive HAL teleop
python main.py navigation                  # navigation closed-loop teleop
python main.py odometry                    # odometry teleop + path plot
python main.py ground_truth_correction     # GTC teleop + before/after plot
python main.py initial_calibration         # two-wall pose calibration
python main.py detect                      # continuous block detection printout
python main.py approach red                # approach a red block until in reach
python main.py in_reach red                # print whether block is in reach
python main.py fsm_to_home red             # find block, grab it, return home
python main.py ekf                         # interactive EKF state inspector
python main.py qr                          # live QR code detection / pose decode
python main.py planner                     # occupancy grid + A* visualisation
python main.py camera_cal                  # offline camera calibration tool
```

## Runtime tuning

Tuning overrides persist across sessions via `runtime.toml` in the working directory.
At startup, any existing `runtime.toml` is loaded before the FSM begins.  During a
REPL session, use `:save` to write the current diffs, `:diff` to preview them, and
`:reset` to discard all in-session changes without touching the file.

```bash
# Example: lower arrival tolerance, save for next run
fp> set ARRIVAL_TOLERANCE_CM 6.0
fp> :diff
  ARRIVAL_TOLERANCE_CM: 8.0 -> 6.0
fp> :save
Saved 1 param(s) to 'runtime.toml'
```

## Running tests

```bash
cd final_project/
python -m pytest tests/ -v          # 152 pass, 1 skip (camera calibration)
bash scripts/verify.sh              # full gate: pytest + py_compile + smoke import
```
