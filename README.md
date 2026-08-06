# Autonomous-Mobile-Hockey-Robot

This project implements a cooperative mobile-hockey task using two DJI RoboMaster EP robots, ROS 2, Python, approximate input-output linearization, and CLF-CBF-QP-based obstacle avoidance.

Robot 4 approaches and acquires a hockey stick, navigates behind the green puck, and passes the puck to Robot 5. Robot 5 independently acquires its stick, moves to a receiving position, receives the pass, repositions behind the puck, and shoots it into the goal.

## Project Objective

This project implements an autonomous mobile hockey robot capable of completing a full hockey task sequence using feedback control, motion planning, and obstacle avoidance. The robot autonomously navigates through the environment, manipulates a hockey stick, interacts with the puck, and performs passing and shooting actions while maintaining safe motion around obstacles.

The navigation and manipulation framework combines the following control methods:

- **Approximate Input–Output Linearization:** Used for differential-drive trajectory tracking, waypoint navigation, and accurate positioning during approach, passing, and shooting tasks.

- **Closed-Loop Feedback Control:** Uses real-time motion-capture data from the VRPN system to continuously estimate the robot pose, correct motion errors, and improve tracking accuracy.

- **CLF-CBF-QP-Based Control:** Generates safe linear and angular velocity commands by combining:
  - A **Control Lyapunov Function (CLF)** to drive the robot toward its target.
  - **Control Barrier Functions (CBFs)** to enforce collision-avoidance and safety constraints.
  - A **Quadratic Program (QP)** to compute control inputs that balance task completion and safe navigation.





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
