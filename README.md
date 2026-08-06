# Autonomous-Mobile-Hockey-Robot

## Project Objective

This project implements an autonomous mobile hockey robot capable of completing a full hockey task sequence using feedback control, motion planning, and obstacle avoidance. The robot autonomously navigates through the environment, picks up a hockey stick, interacts with the puck, and performs passing and shooting actions while maintaining safe motion around obstacles.

The navigation and pick-up framework combines the following control methods:

- **Approximate Input–Output Linearization**
- **Closed-Loop Feedback Control** 
- **CLF-CBF-QP-Based Control:** 
  - **Control Lyapunov Function (CLF)** 
  - **Control Barrier Functions (CBFs)**
  - **Quadratic Program (QP)**



## Task Sequence

### T1 — Stick Acquisition

### T2 — Stick Acquisition

### T3 — Navigate to the Puck

### T4 — Pass and Shoot






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


## Plotting Results
- robot trajectories,
- state evolution,
- linear velocity,
- angular velocity,
- position convergence, and
- obstacle-clearance behavior.

## Repository Structure
| File | Purpose |
|---|---|
| `T1_R4.py` | Robot 4 navigation to and acquisition of the left stick |
| `T1_R5.py` | Robot 5 navigation to and acquisition of the right stick |
| `T3.py` | Robot 4 navigation to a staging point behind the green puck |
| `T4.py` | Cooperative passing and shooting state machine |
| `simulator_env.py` | Mobile-hockey simulation environment |
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

## Course
Developed as an ECE 687 graduate project at the University of Waterloo.
