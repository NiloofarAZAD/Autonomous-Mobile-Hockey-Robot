# wait for robot and puck poses
# navigate to staging point behind green puck
# align with puck-to-goal direction
# stop
# hand control to T4
# defaults:
# robot_id = 5
# puck_topic = '/vrpn_mocap/hockey_puck_green/pose'
# goal_x = 2.0
# goal_y = 0.0
# puck_approach_distance = 0.50
# point_offset = 0.10
import math
import re
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from qpsolvers import solve_qp
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import (
   DurabilityPolicy,
   QoSProfile,
   ReliabilityPolicy,
)

def yaw_from_quaternion(q) -> float:
   """Return planar yaw angle from a quaternion."""
   siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
   cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
   return math.atan2(siny_cosp, cosy_cosp)

def clamp(value: float, minimum: float, maximum: float) -> float:
   """Limit value to [minimum, maximum]."""
   return max(min(value, maximum), minimum)

def wrap_angle(angle: float) -> float:
   """Wrap an angle to [-pi, pi]."""
   return math.atan2(math.sin(angle), math.cos(angle))

class T3Navigate(Node):
   def __init__(self):
       super().__init__('t3_navigate')
       self.last_debug_log_ns = 0
       # ============================================================
       # Parameters
       # ============================================================
       # Green puck target selection
       # Robot selection
       self.declare_parameter('robot_id', 5)
       # Distance from the robot VRPN rigid-body origin to point p.
       # Point p is the point controlled by approximate linearization.
       self.declare_parameter('point_offset', 0.10)
       # Position tolerances
       self.declare_parameter('approach_tolerance', 0.10)
       self.declare_parameter('position_tolerance', 0.05)
       # Final heading control
       self.declare_parameter('heading_tolerance_deg', 5.0)
       self.declare_parameter('heading_gain', 0.40)
       self.declare_parameter('shot_heading_offset_deg', 0.0)
       # Controller gains
       self.declare_parameter('attractive_gain', 0.40)
       # Velocity limits
       self.declare_parameter('max_v', 0.25)
       self.declare_parameter('max_w', 0.25)
       self.declare_parameter('max_cartesian_speed', 0.40)
       # Obstacle dimensions and clearances
       self.declare_parameter('own_robot_radius', 0.15)
       self.declare_parameter('other_robot_radius', 0.15)
       self.declare_parameter('puck_radius', 0.05)
       self.declare_parameter('stick_obstacle_radius', 0.05)
       # A CBF constraint becomes active when surface-to-surface
       # clearance is smaller than obstacle_influence_clearance.
       self.declare_parameter(
           'obstacle_influence_clearance',
           0.60
       )
       # Required safety margin used by the CBF constraints.
       self.declare_parameter('minimum_clearance', 0.08)
       # Stop immediately if estimated surface clearance is below this.
       self.declare_parameter('emergency_clearance', 0.05)
       # Ignore object poses that have not been refreshed recently.
       self.declare_parameter('obstacle_timeout', 0.50)
       # Whether pucks should also be avoided during T3.
       self.declare_parameter('include_pucks_as_obstacles', True)

       # CLF-CBF-QP parameters.
       # Existing navigation, geometry, tolerance, and velocity parameters
       # above are intentionally unchanged.
       self.declare_parameter('clf_rate', 1.0)
       self.declare_parameter('cbf_rate', 2.0)
       self.declare_parameter('clf_slack_weight', 1000.0)
       self.declare_parameter('qp_solver', 'quadprog')

       # Selected green-puck VRPN topic. This target puck is excluded
       # from the CBF obstacle set; other pucks remain obstacles.
       self.declare_parameter(
           'puck_topic',
           '/vrpn_mocap/hockey_puck_green/pose'
       )

       # Known shot destination. theta_shot points from the puck to this goal.
       self.declare_parameter('goal_x', 2.0)
       self.declare_parameter('goal_y', 0.0)

       # T3 stops behind the puck by this distance along -theta_shot.
       self.declare_parameter('puck_approach_distance', 0.50)

       # Geometry-based pre-contact spacing for the attached stick.
       self.declare_parameter('stick_tip_offset_from_point', 0.45)
       self.declare_parameter('t3_puck_gap', 0.23)
       # ============================================================
       # Read parameters
       # ============================================================
       self.robot_id = int(
           self.get_parameter('robot_id').value
       )
       self.point_offset = float(
           self.get_parameter('point_offset').value
       )
       if self.point_offset <= 0.0:
           raise ValueError('point_offset must be greater than zero.')
       self.approach_tolerance = float(
           self.get_parameter('approach_tolerance').value
       )
       self.position_tolerance = float(
           self.get_parameter('position_tolerance').value
       )
       self.heading_tolerance = math.radians(
           float(self.get_parameter('heading_tolerance_deg').value)
       )
       self.k_heading = float(
           self.get_parameter('heading_gain').value
       )
       self.shot_heading_offset = math.radians(
           float(self.get_parameter('shot_heading_offset_deg').value)
       )
       self.k_att = float(
           self.get_parameter('attractive_gain').value
       )
       self.max_v = float(
           self.get_parameter('max_v').value
       )
       self.max_w = float(
           self.get_parameter('max_w').value
       )
       self.max_cartesian_speed = float(
           self.get_parameter('max_cartesian_speed').value
       )
       self.own_robot_radius = float(
           self.get_parameter('own_robot_radius').value
       )
       self.other_robot_radius = float(
           self.get_parameter('other_robot_radius').value
       )
       self.puck_radius = float(
           self.get_parameter('puck_radius').value
       )
       self.stick_obstacle_radius = float(
           self.get_parameter('stick_obstacle_radius').value
       )
       self.obstacle_influence_clearance = float(
           self.get_parameter(
               'obstacle_influence_clearance'
           ).value
       )
       self.minimum_clearance = float(
           self.get_parameter('minimum_clearance').value
       )
       self.emergency_clearance = float(
           self.get_parameter('emergency_clearance').value
       )
       self.obstacle_timeout = float(
           self.get_parameter('obstacle_timeout').value
       )
       self.include_pucks = bool(
           self.get_parameter(
               'include_pucks_as_obstacles'
           ).value
       )

       self.clf_rate = float(
           self.get_parameter('clf_rate').value
       )
       self.cbf_rate = float(
           self.get_parameter('cbf_rate').value
       )
       self.clf_slack_weight = float(
           self.get_parameter('clf_slack_weight').value
       )
       self.qp_solver = str(
           self.get_parameter('qp_solver').value
       ).strip()

       if self.clf_rate <= 0.0:
           raise ValueError('clf_rate must be positive.')
       if self.cbf_rate <= 0.0:
           raise ValueError('cbf_rate must be positive.')
       if self.clf_slack_weight <= 0.0:
           raise ValueError('clf_slack_weight must be positive.')
       if not self.qp_solver:
           raise ValueError('qp_solver cannot be empty.')

       self.puck_topic = str(
           self.get_parameter('puck_topic').value
       )
       self.goal_x = float(
           self.get_parameter('goal_x').value
       )
       self.goal_y = float(
           self.get_parameter('goal_y').value
       )
       self.puck_approach_distance = float(
           self.get_parameter('puck_approach_distance').value
       )
       if self.puck_approach_distance <= 0.0:
           raise ValueError('puck_approach_distance must be positive.')

       self.stick_tip_offset_from_point = float(
           self.get_parameter('stick_tip_offset_from_point').value
       )
       self.t3_puck_gap = float(
           self.get_parameter('t3_puck_gap').value
       )

       if self.stick_tip_offset_from_point <= 0.0:
           raise ValueError(
               'stick_tip_offset_from_point must be positive.'
           )
       if self.t3_puck_gap < 0.0:
           raise ValueError('t3_puck_gap cannot be negative.')

       # Effective T3 staging distance:
       # stick-tip reach + puck radius + desired pre-contact gap.
       self.puck_approach_distance = (
           self.stick_tip_offset_from_point
           + self.puck_radius
           + self.t3_puck_gap
       )
       robot_pose_topic = (
           f'/vrpn_mocap/dji_robot_{self.robot_id}/pose'
       )
       cmd_vel_topic = f'/robot{self.robot_id}/cmd_vel'
       # ============================================================
       # State
       # ============================================================
       self.robot_pose_received = False
       self.puck_pose_received = False
       self.goal_reached = False
       self.goal_reached_time_ns = None
       self.handoff_complete = False
       self.x = 0.0
       self.y = 0.0
       self.theta = 0.0
       self.puck_x = 0.0
       self.puck_y = 0.0
       # Navigation stages: navigate -> align -> complete
       self.navigation_stage = 'navigate'
       # topic -> (x, y, radius, last_update_nanoseconds)
       self.obstacle_poses: Dict[
           str,
           Tuple[float, float, float, int]
       ] = {}
       self.last_waiting_log_ns = 0
       self.last_safety_log_ns = 0
       self.last_qp_log_ns = 0
       # ============================================================
       # QoS
       # ============================================================
       self.best_effort_qos = QoSProfile(
           reliability=ReliabilityPolicy.BEST_EFFORT,
           durability=DurabilityPolicy.VOLATILE,
           depth=1,
       )
       # ============================================================
       # Publisher
       # ============================================================
       self.cmd_pub = self.create_publisher(
           Twist,
           cmd_vel_topic,
           10,
       )
       # ============================================================
       # Subscribers
       # ============================================================
       self.robot_pose_sub = self.create_subscription(
           PoseStamped,
           robot_pose_topic,
           self.robot_pose_callback,
           self.best_effort_qos,
       )
       self.puck_pose_sub = self.create_subscription(
           PoseStamped,
           self.puck_topic,
           self.puck_pose_callback,
           self.best_effort_qos,
       )
       # Subscription objects must remain referenced.
       self.obstacle_subscriptions = []
       self.dynamic_obstacle_subscriptions = []
       self.subscribed_obstacle_topics = set()

       obstacle_specs = self.create_obstacle_specs()
       for topic, obstacle_radius in obstacle_specs:
           subscription = self.create_subscription(
               PoseStamped,
               topic,
               lambda msg,
                      topic_name=topic,
                      radius=obstacle_radius:
                   self.obstacle_pose_callback(
                       msg,
                       topic_name,
                       radius,
                   ),
               self.best_effort_qos,
           )
           self.obstacle_subscriptions.append(subscription)
           self.subscribed_obstacle_topics.add(topic)

       # Discover simulator object topics, including suffixed names.
       self.obstacle_discovery_timer = self.create_timer(
           0.50,
           self.discover_object_obstacle_topics,
       )

       # 20 Hz control loop
       self.timer = self.create_timer(
           0.05,
           self.control_loop,
       )
       self.get_logger().info(
           f'Controlling robot {self.robot_id}'
       )
       self.get_logger().info(
           f'Robot pose topic: {robot_pose_topic}'
       )
       self.get_logger().info(
           f'Command topic: {cmd_vel_topic}'
       )
       self.get_logger().info(
           f'Green puck pose topic: {self.puck_topic}'
       )
       self.get_logger().info(
           f'Controlled-point offset: '
           f'{self.point_offset:.3f} m'
       )
       self.get_logger().info(
           f'Listening to {len(obstacle_specs)} obstacle topics'
       )
       self.get_logger().info(
           'T3 navigation uses approximate linearization with '
           'a CLF-CBF-QP safety filter.'
       )
       self.get_logger().info(
           'Non-target pucks and stick rigid-body obstacle topics are '
           'discovered dynamically, including simulator suffixes such as _2.'
       )
       self.get_logger().info(
           f'Target green puck excluded from CBF obstacles: {self.puck_topic}'
       )
       self.get_logger().info(
           f'T3 geometry-based staging distance: '
           f'{self.puck_approach_distance:.3f} m '
           f'(stick tip {self.stick_tip_offset_from_point:.3f} m + '
           f'puck radius {self.puck_radius:.3f} m + '
           f'gap {self.t3_puck_gap:.3f} m)'
       )
   # ================================================================
   # Topic configuration
   # ================================================================
   def create_obstacle_specs(
       self
   ) -> list[Tuple[str, float]]:
       """Return statically known obstacle topics."""
       specs: list[Tuple[str, float]] = []

       # Other RoboMaster robots are always known by ID.
       for robot_id in range(1, 11):
           if robot_id == self.robot_id:
               continue

           specs.append((
               f'/vrpn_mocap/dji_robot_{robot_id}/pose',
               self.other_robot_radius,
           ))

       # Pucks and stick rigid bodies are discovered dynamically because
       # the simulator may append suffixes such as "_2".
       return specs

   def discover_object_obstacle_topics(self) -> None:
       """
       Discover all tracked hockey pucks and stick rigid bodies.

       Examples handled:
           hockey_puck_blue
           hockey_puck_blue_2
           hockey_sticks_1
           hockey_sticks_1_2
           hockey_stick_1
       """
       object_pattern = re.compile(
           r'^/vrpn_mocap/'
           r'(hockey_puck_[^/]+|hockey_sticks?_[^/]+)'
           r'/pose$'
       )

       for topic_name, topic_types in (
           self.get_topic_names_and_types()
       ):
           if (
               'geometry_msgs/msg/PoseStamped'
               not in topic_types
           ):
               continue

           if topic_name in self.subscribed_obstacle_topics:
               continue

           # The selected green puck is the T3 target, so it must not be
           # treated as an obstacle. Other pucks and sticks remain obstacles.
           if topic_name == self.puck_topic:
               continue

           match = object_pattern.match(topic_name)
           if match is None:
               continue

           object_name = match.group(1)

           if object_name.startswith('hockey_puck_'):
               if not self.include_pucks:
                   continue
               obstacle_radius = self.puck_radius
           else:
               obstacle_radius = self.stick_obstacle_radius

           subscription = self.create_subscription(
               PoseStamped,
               topic_name,
               lambda msg,
                      discovered_topic=topic_name,
                      radius=obstacle_radius:
                   self.obstacle_pose_callback(
                       msg,
                       discovered_topic,
                       radius,
                   ),
               self.best_effort_qos,
           )

           self.dynamic_obstacle_subscriptions.append(
               subscription
           )
           self.subscribed_obstacle_topics.add(topic_name)

           self.get_logger().info(
               f'Discovered obstacle topic: {topic_name}'
           )

   # ================================================================
   # Callbacks
   # ================================================================
   def robot_pose_callback(
       self,
       msg: PoseStamped
   ) -> None:
       self.x = msg.pose.position.x
       self.y = msg.pose.position.y
       self.theta = yaw_from_quaternion(
           msg.pose.orientation
       )
       self.robot_pose_received = True
   def puck_pose_callback(
       self,
       msg: PoseStamped
   ) -> None:
       self.puck_x = msg.pose.position.x
       self.puck_y = msg.pose.position.y
       self.puck_pose_received = True

   def obstacle_pose_callback(
       self,
       msg: PoseStamped,
       topic_name: str,
       obstacle_radius: float,
   ) -> None:
       now_ns = self.get_clock().now().nanoseconds
       self.obstacle_poses[topic_name] = (
           msg.pose.position.x,
           msg.pose.position.y,
           obstacle_radius,
           now_ns,
       )

   # ================================================================
   # T3 target geometry
   # ================================================================
   def calculate_shot_heading(self) -> float:
       """Return theta_shot from the selected puck toward the known goal."""
       dx = self.goal_x - self.puck_x
       dy = self.goal_y - self.puck_y
       if math.hypot(dx, dy) < 1e-6:
           raise ValueError('The goal position cannot equal the puck position.')
       return math.atan2(dy, dx)

   def calculate_t3_target(self) -> Tuple[float, float, float]:
       """
       Compute the staging point behind the puck:

           p_T3 = p_puck - d_approach [cos(theta_shot), sin(theta_shot)]^T
       """
       theta_shot = self.calculate_shot_heading()
       target_x = (
           self.puck_x
           - self.puck_approach_distance * math.cos(theta_shot)
       )
       target_y = (
           self.puck_y
           - self.puck_approach_distance * math.sin(theta_shot)
       )
       return target_x, target_y, theta_shot

   # ================================================================
   # CLF-CBF-QP navigation
   # ================================================================
   def active_obstacles(
       self,
       px: float,
       py: float,
   ) -> Tuple[List[Tuple[float, float, float]], bool]:
       """
       Return fresh obstacles close enough to require a CBF constraint.

       Distances are evaluated from the controlled point p because the
       approximate-linearized dynamics are p_dot = [ux, uy].
       """
       active: List[Tuple[float, float, float]] = []
       emergency_stop_required = False

       now_ns = self.get_clock().now().nanoseconds
       timeout_ns = int(self.obstacle_timeout * 1e9)

       for (
           obstacle_x,
           obstacle_y,
           obstacle_radius,
           update_time_ns,
       ) in self.obstacle_poses.values():
           if now_ns - update_time_ns > timeout_ns:
               continue

           point_distance = math.hypot(
               px - obstacle_x,
               py - obstacle_y,
           )

           # Inflate the circular safety envelope by point_offset so the
           # chassis body is protected while the front point is controlled.
           combined_radius = (
               self.own_robot_radius
               + obstacle_radius
               + self.point_offset
           )

           surface_clearance = (
               point_distance
               - combined_radius
           )

           if surface_clearance <= self.emergency_clearance:
               emergency_stop_required = True

           if (
               surface_clearance
               < self.obstacle_influence_clearance
           ):
               active.append((
                   obstacle_x,
                   obstacle_y,
                   obstacle_radius,
               ))

       return active, emergency_stop_required

   def rotation_is_safe(self) -> bool:
       """Check the full chassis/controlled-point rotation footprint."""
       now_ns = self.get_clock().now().nanoseconds
       timeout_ns = int(self.obstacle_timeout * 1e9)

       for (
           obstacle_x,
           obstacle_y,
           obstacle_radius,
           update_time_ns,
       ) in self.obstacle_poses.values():
           if now_ns - update_time_ns > timeout_ns:
               continue

           center_distance = math.hypot(
               self.x - obstacle_x,
               self.y - obstacle_y,
           )

           required_distance = (
               self.own_robot_radius
               + self.point_offset
               + obstacle_radius
               + self.minimum_clearance
           )

           if center_distance <= required_distance:
               return False

       return True

   def solve_clf_cbf_qp(
       self,
       px: float,
       py: float,
       target_x: float,
       target_y: float,
       nominal_ux: float,
       nominal_uy: float,
   ) -> Tuple[float, float, bool, bool]:
       """
       Solve the CLF-CBF quadratic program.

       Decision variables:
           z = [ux, uy, delta]

       ux and uy are the optimized Cartesian velocities of the
       approximate-linearization point p. Delta is the nonnegative CLF
       relaxation variable. CBF constraints remain hard constraints.
       """
       obstacles, emergency_stop_required = (
           self.active_obstacles(px, py)
       )

       if emergency_stop_required:
           return 0.0, 0.0, False, True

       # Objective:
       #   ||u - u_nominal||^2
       #   + clf_slack_weight * delta^2
       p_matrix = np.diag([
           2.0,
           2.0,
           2.0 * self.clf_slack_weight,
       ])

       q_vector = np.array([
           -2.0 * nominal_ux,
           -2.0 * nominal_uy,
           0.0,
       ])

       g_rows = []
       h_values = []

       # ------------------------------------------------------------
       # CLF: V = 0.5 * ||p - p_target||^2
       #
       #   V_dot <= -clf_rate * V + delta
       #
       # With p_dot = u:
       #   e^T u - delta <= -clf_rate * V
       # ------------------------------------------------------------
       clf_error_x = px - target_x
       clf_error_y = py - target_y

       lyapunov_value = 0.5 * (
           clf_error_x ** 2
           + clf_error_y ** 2
       )

       g_rows.append([
           clf_error_x,
           clf_error_y,
           -1.0,
       ])
       h_values.append(
           -self.clf_rate * lyapunov_value
       )

       # delta >= 0  ->  -delta <= 0
       g_rows.append([0.0, 0.0, -1.0])
       h_values.append(0.0)

       cos_theta = math.cos(self.theta)
       sin_theta = math.sin(self.theta)

       # ------------------------------------------------------------
       # CBF constraints for the approximate-linearized point.
       #
       # h_i = ||p - p_obstacle_i||^2 - d_safe_i^2
       # h_dot = 2 (p - p_obstacle_i)^T u
       #
       # point_offset is included in d_safe so the chassis footprint is
       # conservatively protected while p is controlled.
       # ------------------------------------------------------------
       for (
           obstacle_x,
           obstacle_y,
           obstacle_radius,
       ) in obstacles:
           dx = px - obstacle_x
           dy = py - obstacle_y

           safe_distance = (
               self.own_robot_radius
               + obstacle_radius
               + self.point_offset
               + self.minimum_clearance
           )

           barrier_value = (
               dx ** 2
               + dy ** 2
               - safe_distance ** 2
           )

           # h_dot + cbf_rate * h >= 0
           # -2*dx*ux - 2*dy*uy <= cbf_rate*h
           g_rows.append([
               -2.0 * dx,
               -2.0 * dy,
               0.0,
           ])
           h_values.append(
               self.cbf_rate * barrier_value
           )

       # ------------------------------------------------------------
       # Input constraints are included inside the QP so the optimizer
       # accounts for the existing v and omega limits.
       # ------------------------------------------------------------

       # v = cos(theta) ux + sin(theta) uy
       g_rows.extend([
           [cos_theta, sin_theta, 0.0],
           [-cos_theta, -sin_theta, 0.0],
       ])
       h_values.extend([
           self.max_v,
           self.max_v,
       ])

       # omega = (-sin(theta) ux + cos(theta) uy) / point_offset
       omega_ux = -sin_theta / self.point_offset
       omega_uy = cos_theta / self.point_offset

       g_rows.extend([
           [omega_ux, omega_uy, 0.0],
           [-omega_ux, -omega_uy, 0.0],
       ])
       h_values.extend([
           self.max_w,
           self.max_w,
       ])

       # Cartesian component limits. The existing exact Euclidean-speed
       # limit is still checked after solving.
       g_rows.extend([
           [1.0, 0.0, 0.0],
           [-1.0, 0.0, 0.0],
           [0.0, 1.0, 0.0],
           [0.0, -1.0, 0.0],
       ])
       h_values.extend([
           self.max_cartesian_speed,
           self.max_cartesian_speed,
           self.max_cartesian_speed,
           self.max_cartesian_speed,
       ])

       g_matrix = np.asarray(g_rows, dtype=float)
       h_vector = np.asarray(h_values, dtype=float)

       # Try the configured solver first, then common installed backends.
       solver_candidates = []
       for solver_name in (
           self.qp_solver,
           'quadprog',
           'cvxopt',
           'osqp',
       ):
           if solver_name not in solver_candidates:
               solver_candidates.append(solver_name)

       solution = None

       for solver_name in solver_candidates:
           try:
               solution = solve_qp(
                   P=p_matrix,
                   q=q_vector,
                   G=g_matrix,
                   h=h_vector,
                   solver=solver_name,
               )
           except Exception:
               solution = None

           if (
               solution is not None
               and len(solution) >= 3
               and np.all(np.isfinite(solution))
           ):
               break

       if solution is None:
           return 0.0, 0.0, False, False

       ux = float(solution[0])
       uy = float(solution[1])

       # Preserve the original exact Cartesian-speed bound.
       cartesian_speed = math.hypot(ux, uy)

       if cartesian_speed > self.max_cartesian_speed:
           scale = (
               self.max_cartesian_speed
               / cartesian_speed
           )
           ux *= scale
           uy *= scale

       return ux, uy, True, False

   # ================================================================
   # Logging helpers
   # ================================================================
   def log_periodically(
       self,
       message: str,
       attribute_name: str,
       period_seconds: float = 2.0,
   ) -> None:
       now_ns = self.get_clock().now().nanoseconds
       previous_ns = getattr(self, attribute_name)
       if now_ns - previous_ns >= int(period_seconds * 1e9):
           self.get_logger().warning(message)
           setattr(self, attribute_name, now_ns)
   # ================================================================
   # Controller
   # ================================================================

   def control_loop(self) -> None:
       # Once the T3 staging point and shot heading have been reached, hold the chassis
       # stopped for 0.5 s, then shut T1 down so T2 can take control.
       if self.goal_reached:
           self.stop_robot()
           if self.goal_reached_time_ns is not None:
               elapsed_ns = (
                   self.get_clock().now().nanoseconds
                   - self.goal_reached_time_ns
               )
               if elapsed_ns >= int(0.5e9):
                   self.get_logger().info(
                       'T3 handoff complete. Releasing chassis control.'
                   )
                   self.handoff_complete = True
           return
       if not self.robot_pose_received:
           self.stop_robot()
           self.log_periodically(
               'Waiting for robot VRPN pose...',
               'last_waiting_log_ns',
           )
           return
       if not self.puck_pose_received:
           self.stop_robot()
           self.log_periodically(
               'Waiting for green-puck VRPN pose...',
               'last_waiting_log_ns',
           )
           return
       # Controlled point p in front of the robot rigid-body origin.
       px = (
           self.x
           + self.point_offset * math.cos(self.theta)
       )
       py = (
           self.y
           + self.point_offset * math.sin(self.theta)
       )

       target_x, target_y, theta_shot = self.calculate_t3_target()
       error_x = target_x - px
       error_y = target_y - py
       position_error = math.hypot(error_x, error_y)
       # ------------------------------------------------------------
       # Debug output
       # ------------------------------------------------------------
       now_ns = self.get_clock().now().nanoseconds
       if now_ns - self.last_debug_log_ns >= int(0.5e9):
           self.get_logger().info(
               f'Stage={self.navigation_stage}, '
               f'robot=({self.x:.3f}, {self.y:.3f}), '
               f'point_p=({px:.3f}, {py:.3f}), '
               f'puck=({self.puck_x:.3f}, {self.puck_y:.3f}), '
               f'theta_shot={math.degrees(theta_shot):.1f} deg, '
               f'target=({target_x:.3f}, {target_y:.3f}), '
               f'error={position_error:.3f} m'
           )
           self.last_debug_log_ns = now_ns

       # ------------------------------------------------------------
       # Reach the staging point behind the puck, then align for T4.
       # ------------------------------------------------------------
       if (
           self.navigation_stage == 'navigate'
           and position_error <= self.position_tolerance
       ):
           self.stop_robot()
           self.navigation_stage = 'align'
           self.get_logger().info(
               'T3 staging point reached. Aligning with puck-to-goal direction.'
           )
           return

       # ------------------------------------------------------------
       # Final shot-heading alignment only
       # ------------------------------------------------------------
       if self.navigation_stage == 'align':
           if not self.rotation_is_safe():
               self.stop_robot()
               self.log_periodically(
                   'Final heading alignment is paused because an '
                   'obstacle intersects the rotation safety footprint.',
                   'last_safety_log_ns',
                   1.0,
               )
               return

           desired_theta = wrap_angle(
               theta_shot + self.shot_heading_offset
           )
           theta_error = wrap_angle(
               desired_theta - self.theta
           )
           if abs(theta_error) <= self.heading_tolerance:
               self.goal_reached = True
               self.goal_reached_time_ns = (
                   self.get_clock().now().nanoseconds
               )
               self.stop_robot()
               self.get_logger().info(
                   'T3 complete: staging point reached and shot heading '
                   'aligned. Holding v=0 and omega=0 for 0.5 s '
                   'before handoff.'
               )
               return
           # Keep translation locked to zero while theta is corrected.
           omega = clamp(
               self.k_heading * theta_error,
               -self.max_w,
               self.max_w,
           )
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = omega
           self.cmd_pub.publish(cmd)
           return

       # Nominal stabilizing velocity for the controlled point p.
       # The original attractive gain is preserved.
       nominal_ux = self.k_att * error_x
       nominal_uy = self.k_att * error_y

       (
           ux,
           uy,
           qp_success,
           emergency_stop_required,
       ) = self.solve_clf_cbf_qp(
           px=px,
           py=py,
           target_x=target_x,
           target_y=target_y,
           nominal_ux=nominal_ux,
           nominal_uy=nominal_uy,
       )

       if emergency_stop_required:
           self.stop_robot()
           self.log_periodically(
               'Emergency stop: an obstacle is inside '
               'the emergency clearance.',
               'last_safety_log_ns',
               1.0,
           )
           return

       if not qp_success:
           self.stop_robot()
           self.log_periodically(
               'CLF-CBF-QP failed to return a feasible command. '
               'The robot is being held stopped.',
               'last_qp_log_ns',
               1.0,
           )
           return
       # Approximate linearization:
       #
       # [v, omega]^T = L^-1(l) R^T(theta) p_dot
       v = (
           math.cos(self.theta) * ux
           + math.sin(self.theta) * uy
       )
       omega = (
           -math.sin(self.theta) * ux
           + math.cos(self.theta) * uy
       ) / self.point_offset
       v = clamp(v, -self.max_v, self.max_v)
       omega = clamp(
           omega,
           -self.max_w,
           self.max_w,
       )
       cmd = Twist()
       cmd.linear.x = v
       cmd.angular.z = omega
       self.cmd_pub.publish(cmd)
   def stop_robot(self) -> None:
       # Explicitly command a complete chassis stop.
       cmd = Twist()
       cmd.linear.x = 0.0
       cmd.linear.y = 0.0
       cmd.linear.z = 0.0
       cmd.angular.x = 0.0
       cmd.angular.y = 0.0
       cmd.angular.z = 0.0
       self.cmd_pub.publish(cmd)

def main(args=None):
   rclpy.init(args=args)
   node = T3Navigate()
   try:
       # Use spin_once so the process can exit cleanly when the
       # handoff_complete flag is set by the control loop.
       while rclpy.ok() and not node.handoff_complete:
           rclpy.spin_once(node, timeout_sec=0.1)
   except KeyboardInterrupt:
       pass
   finally:
       if rclpy.ok():
           try:
               node.stop_robot()
           except Exception:
               pass
       node.destroy_node()
       if rclpy.ok():
           rclpy.shutdown()

if __name__ == '__main__':
   main()