# Autonomous-Mobile-Hockey-Robot

This project implements a cooperative mobile-hockey task using two DJI RoboMaster EP robots, ROS 2, Python, approximate input-output linearization, and CLF-CBF-QP-based obstacle avoidance.

Robot 4 approaches and acquires a hockey stick, navigates behind the green puck, and passes the puck to Robot 5. Robot 5 independently acquires its stick, moves to a receiving position, receives the pass, repositions behind the puck, and shoots it into the goal.

## Project Objective

The objective is to coordinate two unicycle-type mobile robots to complete the following sequence:

1. Robot 4 navigates to and acquires the left hockey stick.
2. Robot 5 navigates to and acquires the right hockey stick.
3. Robot 4 navigates to a staging position behind the green puck.
4. Robot 5 moves to a receiving position near the goal.
5. Robot 4 passes the puck to Robot 5.
6. Robot 5 detects the received puck.
7. Robot 5 moves behind the puck relative to the goal.
8. Robot 5 aligns with the goal and shoots.
9. Both robots stop safely.

During navigation, the robots avoid other robots, non-target pucks, and non-target sticks.

## Main Features

- Cooperative control of two DJI RoboMaster EP robots
- ROS 2 Humble implementation in Python
- VRPN motion-capture pose feedback
- Unicycle robot modeling
- Virtual controlled-point formulation
- Approximate input-output linearization
- CLF-CBF-QP navigation and obstacle avoidance
- Quadratic-program-based velocity optimization
- Automatic stick-side selection
- Simulator stick attachment
- Geometry-based puck staging
- Closed-loop passing and shooting strokes
- State-machine-based task coordination
- Emergency stopping and velocity constraints
- CSV trajectory and control-data logging
- Plotting utilities for controller evaluation

## Task Sequence

### T1 — Stick Acquisition

`T1_R4.py` and `T1_R5.py` control Robots 4 and 5 independently.

Each robot:

- subscribes to its VRPN pose,
- subscribes to the selected stick pose,
- computes a pickup point from the stick pose and local offsets,
- first moves to a pre-pickup approach point,
- advances to the pickup point,
- performs final heading alignment,
- sends a simulator stick-attachment command, and
- saves its motion data.

Robot 4 uses the left stick and Robot 5 uses the right stick.

### T3 — Navigate to the Puck

`T3.py` controls Robot 4.

The controller:

- tracks the green puck,
- computes the puck-to-goal direction,
- calculates a staging point behind the puck,
- accounts for stick-tip length, puck radius, and pre-contact spacing,
- navigates Robot 4 to the staging point,
- aligns Robot 4 with the puck-to-goal direction, and
- stops before handing control to T4.

### T4 — Pass and Shoot

`T4.py` coordinates both robots through a state machine.

The implemented states are:

```text
WAIT_FOR_POSES
    ↓
MOVE_R5_TO_RECEIVE
    ↓
ALIGN_R5_TO_RECEIVE
    ↓
MOVE_R4_TO_PASS_STAGE
    ↓
ALIGN_R4_TO_PASS
    ↓
PASS_WITH_R4
    ↓
WAIT_FOR_PUCK_AT_R5
    ↓
MOVE_R5_TO_GOAL_STAGE
    ↓
ALIGN_R5_TO_GOAL
    ↓
SHOOT_WITH_R5
    ↓
COMPLETE
```

The T4 controller includes:

- a fixed receiving-pose calculation,
- Robot 4 pass alignment,
- a closed-loop passing stroke,
- simulated puck motion with drag,
- puck-reception detection,
- Robot 5 goal-stage positioning,
- final goal alignment, and
- a closed-loop shooting stroke.

## Repository Structure


### File Descriptions

| File | Purpose |
|---|---|
| `T1_R4.py` | Robot 4 navigation to and acquisition of the left stick |
| `T1_R5.py` | Robot 5 navigation to and acquisition of the right stick |
| `T3.py` | Robot 4 navigation to a staging point behind the green puck |
| `T4.py` | Cooperative passing and shooting state machine |
| `simulator.py` | Mobile-hockey simulation environment |
| `plot_trajectory.py` | Robot \(x\)-\(y\) trajectory visualization |
| `plot_t_states.py` | Robot state plots \(x(t)\), \(y(t)\), and \(\theta(t)\) |
| `plot_controls.py` | Linear- and angular-velocity plots |
| `plot_position_error.py` | Position-error-versus-time plot |
| `plot_obstacle_clearance.py` | Obstacle-clearance-versus-time plot |

## Requirements

The project was developed for:

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10 or later
- NumPy
- qpsolvers
- at least one supported QP backend
- Matplotlib
- ROS 2 geometry and standard message packages
- VRPN motion-capture topics or the provided simulator

## Plotting Results

Generate the available evaluation plots using:

```bash
python3 plot_trajectory.py
python3 plot_t_states.py
python3 plot_controls.py
python3 plot_position_error.py
python3 plot_obstacle_clearance.py
```

The plots can be used to analyze:

- robot trajectories,
- state evolution,
- linear velocity,
- angular velocity,
- position convergence, and
- obstacle-clearance behavior.

## Safety Behavior

The controllers include several safety mechanisms:

- CBF collision-avoidance constraints
- configurable robot and obstacle radii
- minimum-clearance constraints
- obstacle-influence regions
- stale-pose rejection
- emergency stopping
- velocity saturation
- safe final-rotation checks
- QP failure stopping
- target-object exclusion during deliberate contact

The target stick or puck is excluded from normal obstacle avoidance only when required to complete the corresponding interaction.

## Course

Developed as an ECE 687 graduate project at the University of Waterloo.
