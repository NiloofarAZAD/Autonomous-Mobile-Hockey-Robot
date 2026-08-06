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

**T1 — Navigating to a Known Stick Pick-Up Location**

1. Wait for the robot and the selected stick VRPN poses.
2. Navigate to an approach point behind the stick.
3. Move forward to the calculated stick pick-up point.
4. Stop translation and align the robot with the stick.
5. Hold the robot in the stopped position briefly, then hand control to T2.

**T2 — Picking Up a Stick**

1. Move the arm upward to the stick height.
2. Open the gripper.
3. Move the arm forward toward the stick.
4. Close the gripper and wait for a secure grip.
5. Lift the arm with the stick.
6. Move the robot backward approximately `0.40 m`, stop, and complete T2.

**T3 — Navigating to a Known Puck Location**

1. Wait for the Robot 4 and green-puck poses.
2. Calculate an approach point near the known green-puck position.
3. Navigate Robot 4 to the approach point while avoiding obstacles.
4. Move Robot 4 to the final staging position beside or behind the puck.
5. Stop Robot 4 and hand control to T4.

**T4 — Passing and Shooting**

1. Wait for the Robot 4, Robot 5, and green-puck poses.
2. Move Robot 5 to the receiving location.
3. Align Robot 5 for the incoming pass.
4. Align Robot 4 and its stick with the puck and Robot 5.
5. Robot 4 passes the puck to Robot 5.
6. Confirm that Robot 5 has received the puck.
7. Move Robot 5 behind the puck relative to the known goal location.
8. Align Robot 5 with the goal.
9. Robot 5 shoots the puck into the goal.
10. Stop both robots and complete the task.

**T5 — Avoiding Obstacles Placed at Known Locations**

Obstacle avoidance is applied within the T1, T3, and T4 tasks.






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

## Project Structure
## Project Structure

The project was developed and executed in the **RoboHub** environment using ROS 2 packages and the RoboMaster simulator.

- **ROS 2 Controller Package:**
  [`EP_HOCKEY_CONTROLLER`]

- **RoboMaster Simulation Package:**
  [`multi_robomaster_ros_sim`]

only this

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
