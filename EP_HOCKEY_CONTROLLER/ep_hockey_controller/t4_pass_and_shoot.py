#Robot 5 moves to the receiving position.
#Robot 5 aligns for the incoming pass.
#Robot 4 moves to the green puck.
#Robot 4 completes its forward contact adjustment.
#Robot 4 performs the preload move.
#Robot 4 passes the puck toward Robot 5.
#The controller confirms that the puck moved toward Robot 5 and settled.
#Robot 5 rotates and moves behind the received puck relative to the goal.
#Robot 5 moves to the receiving position.
#Robot 5 aligns for the incoming pass.
#Robot 4 moves to the green puck.
#Robot 4 completes its forward contact adjustment.
#Robot 4 performs the preload.
#Robot 4 passes the puck toward Robot 5.
#The controller confirms that the puck moved toward Robot 5 and settled.
#Robot 5 rotates and moves behind the received puck relative to the goal.



import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from rclpy.parameter import Parameter
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from qpsolvers import solve_qp
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def yaw_from_quaternion(q) -> float:
    """Return planar yaw from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


@dataclass
class RobotState:
    robot_id: int
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    pose_received: bool = False


@dataclass
class ObstacleState:
    x: float
    y: float
    radius: float
    update_time_ns: int
    topic: str


class T4PassAndShoot(Node):
    """Coordinate Robot 4 passing to Robot 5, then Robot 5 shooting."""

    # State-machine names
    WAIT_FOR_POSES = 'wait_for_poses'
    MOVE_R5_TO_RECEIVE = 'move_r5_to_receive'
    ALIGN_R5_TO_RECEIVE = 'align_r5_to_receive'
    MOVE_R4_TO_PASS_STAGE = 'move_r4_to_pass_stage'
    ALIGN_R4_TO_PASS = 'align_r4_to_pass'
    BACKSWING_R4 = 'backswing_r4'
    PRECONTACT_R4 = 'precontact_r4'
    MANUAL_FORWARD_R4 = 'manual_forward_r4'
    PASS_WITH_R4 = 'pass_with_r4'
    WAIT_FOR_PUCK_AT_R5 = 'wait_for_puck_at_r5'
    MOVE_R5_TO_GOAL_STAGE = 'move_r5_to_goal_stage'
    ALIGN_R5_TO_GOAL = 'align_r5_to_goal'
    SHOOT_WITH_R5 = 'shoot_with_r5'
    COMPLETE = 'complete'

    def __init__(self) -> None:
        super().__init__('t4_pass_and_shoot')

        # ============================================================
        # Robot and target configuration
        # ============================================================
        self.declare_parameter('passer_robot_id', 4)
        self.declare_parameter('receiver_robot_id', 5)
        self.declare_parameter(
            'puck_topic',
            '/vrpn_mocap/hockey_puck_green/pose',
        )
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 0.0)

        # Receiver position relative to puck-to-goal geometry.
        self.declare_parameter('receiver_backoff', 0.70)
        self.declare_parameter('receiver_right_offset', 0.60)
        self.declare_parameter('receiver_heading_offset_deg', 0.0)

        # Prevent a large circular path when Robot 5 begins T4 with a
        # heading inherited from T1/T3.
        self.declare_parameter(
            'receiver_departure_heading_tolerance_deg',
            12.0,
        )
        self.declare_parameter(
            'receiver_departure_max_w',
            0.15,
        )

        # Prevent a large circular path when Robot 4 begins moving
        # toward its pass-staging point with an unfavorable heading.
        self.declare_parameter(
            'robot4_departure_heading_tolerance_deg',
            10.0,
        )
        self.declare_parameter(
            'robot4_departure_max_w',
            0.20,
        )

        # ============================================================
        # T1/T3 controller parameters
        # ============================================================
        self.declare_parameter('point_offset', 0.10)
        self.declare_parameter('position_tolerance', 0.05)
        self.declare_parameter('heading_tolerance_deg', 5.0)
        self.declare_parameter('heading_gain', 0.40)
        self.declare_parameter('attractive_gain', 0.40)

        self.declare_parameter('max_v', 0.25)
        self.declare_parameter('max_w', 0.25)
        self.declare_parameter('max_cartesian_speed', 0.40)

        self.declare_parameter('own_robot_radius', 0.15)
        self.declare_parameter('other_robot_radius', 0.15)
        self.declare_parameter('puck_radius', 0.05)
        self.declare_parameter('stick_obstacle_radius', 0.05)
        self.declare_parameter('minimum_clearance', 0.08)
        self.declare_parameter('emergency_clearance', 0.05)
        self.declare_parameter('obstacle_influence_clearance', 0.60)
        self.declare_parameter('obstacle_timeout', 0.50)
        self.declare_parameter('include_pucks_as_obstacles', True)

        self.declare_parameter('clf_rate', 1.0)
        self.declare_parameter('cbf_rate', 2.0)
        self.declare_parameter('clf_slack_weight', 1000.0)
        self.declare_parameter('qp_solver', 'quadprog')

        # ============================================================
        # Stick and contact geometry
        # ============================================================
        self.declare_parameter('stick_tip_offset_from_point', 0.45)
        self.declare_parameter('precontact_gap', 0.10)
        # Robot-4-only precision values for the pass.  These do not change
        # Robot 5's contact geometry or the general navigation tolerance.
        self.declare_parameter('robot4_tip_precontact_gap', 0.015)
        self.declare_parameter('robot4_pass_stage_tolerance', 0.012)
        # A short physical calibration advance after alignment. This closes
        # the remaining simulator/model mismatch without changing the main
        # CLF-CBF-QP staging target or Robot 5 behavior.
        # Robot-4-only adaptive contact closing.  The simulator's visible
        # stick reach is shorter than the geometric marker-to-tip model, so
        # target_gap is intentionally negative: the modeled tip must pass a
        # few centimetres into the puck before the visible stick contacts it.
        self.declare_parameter('robot4_precontact_advance_distance', 0.160)
        self.declare_parameter('robot4_precontact_speed', 0.030)
        self.declare_parameter('robot4_precontact_timeout', 7.0)
        # Explicit final forward nudge before the fast shot.
        self.declare_parameter('robot4_manual_forward_distance', 0.060)
        self.declare_parameter('robot4_manual_forward_speed', 0.025)
        self.declare_parameter('robot4_manual_forward_timeout', 3.5)
        # Accept completion when only a few millimetres remain. The
        # controller intentionally slows near the target, so requiring the
        # exact full distance can create a false timeout.
        self.declare_parameter(
            'robot4_manual_forward_completion_tolerance',
            0.008,
        )
        self.declare_parameter('robot4_manual_forward_heading_gate_deg', 12.0)
        self.declare_parameter('robot4_precontact_heading_gate_deg', 12.0)
        self.declare_parameter('robot4_precontact_target_gap', -0.040)
        self.declare_parameter('robot4_precontact_lateral_tolerance', 0.020)
        self.declare_parameter('robot4_precontact_slow_zone', 0.050)

        # Closed-loop contact stroke parameters.
        self.declare_parameter('pass_stroke_distance', 0.35)
        self.declare_parameter('shot_stroke_distance', 0.40)
        self.declare_parameter('contact_max_v', 0.10)
        self.declare_parameter('contact_max_w', 0.12)
        self.declare_parameter('contact_timeout', 4.0)

        # Robot 4 rotational-pass parameters.
        # The stick angle is measured from the robot's forward x-axis.
        # swing_direction: +1 = counterclockwise, -1 = clockwise.
        # Default is clockwise so Robot 4 first backs up in the opposite
        # direction and then swings toward Robot 5.
        self.declare_parameter('robot4_stick_angle_offset_deg', 36.0)
        self.declare_parameter('robot4_swing_direction', 1)
        # Flip Robot 4 to the opposite side of the puck-centered
        # diameter without changing the swing direction.
        self.declare_parameter('robot4_flip_stage_diameter', False)
        self.declare_parameter('robot4_backswing_angle_deg', 18.0)
        self.declare_parameter('robot4_follow_through_angle_deg', 18.0)
        self.declare_parameter('robot4_backswing_max_w', 0.30)
        self.declare_parameter('robot4_swing_start_w', 1.00)
        self.declare_parameter('robot4_swing_max_w', 1.00)
        self.declare_parameter(
            'robot4_swing_angular_acceleration',
            2.00,
        )
        # Reach maximum angular velocity within this fraction of the
        # backswing-return angle. With 0.35 and a 20 deg backswing, maximum
        # speed is reached after about 7 deg, well before puck impact.
        self.declare_parameter(
            'robot4_preimpact_accel_fraction',
            0.35,
        )
        self.declare_parameter('robot4_swing_timeout', 2.5)

        # Extra separation used only for Robot 4's rotational pass.
        # During the fast return swing, a small forward velocity closes
        # this additional gap so the stick reaches the puck.
        self.declare_parameter(
            'robot4_extra_stage_clearance',
            0.02,
        )

        self.declare_parameter(
            'robot4_pass_stage_distance',
            0.35,
        )
        self.declare_parameter(
            'robot4_swing_forward_speed',
            0.010,
        )

        # Puck-reception detection.
        self.declare_parameter('puck_receive_tolerance', 0.25)
        self.declare_parameter('puck_receive_required_cycles', 5)
        self.declare_parameter('puck_motion_epsilon', 0.015)

        # Optional topic exclusions, useful for attached rigid bodies.
        self.declare_parameter(
            'ignored_obstacle_topics',
            Parameter.Type.STRING_ARRAY,
        )

        # ============================================================
        # Read and validate parameters
        # ============================================================
        passer_id = int(self.get_parameter('passer_robot_id').value)
        receiver_id = int(self.get_parameter('receiver_robot_id').value)
        if passer_id == receiver_id:
            raise ValueError(
                'passer_robot_id and receiver_robot_id must differ.'
            )

        self.robot4 = RobotState(robot_id=passer_id)
        self.robot5 = RobotState(robot_id=receiver_id)

        self.puck_topic = str(self.get_parameter('puck_topic').value)
        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)

        self.receiver_backoff = float(
            self.get_parameter('receiver_backoff').value
        )
        self.receiver_right_offset = float(
            self.get_parameter('receiver_right_offset').value
        )
        self.receiver_heading_offset = math.radians(
            float(self.get_parameter('receiver_heading_offset_deg').value)
        )
        self.receiver_departure_heading_tolerance = math.radians(
            float(
                self.get_parameter(
                    'receiver_departure_heading_tolerance_deg'
                ).value
            )
        )
        self.receiver_departure_max_w = float(
            self.get_parameter(
                'receiver_departure_max_w'
            ).value
        )

        self.robot4_departure_heading_tolerance = math.radians(
            float(
                self.get_parameter(
                    'robot4_departure_heading_tolerance_deg'
                ).value
            )
        )
        self.robot4_departure_max_w = float(
            self.get_parameter(
                'robot4_departure_max_w'
            ).value
        )

        self.point_offset = float(self.get_parameter('point_offset').value)
        self.position_tolerance = float(
            self.get_parameter('position_tolerance').value
        )
        self.heading_tolerance = math.radians(
            float(self.get_parameter('heading_tolerance_deg').value)
        )
        self.k_heading = float(self.get_parameter('heading_gain').value)
        self.k_att = float(self.get_parameter('attractive_gain').value)

        self.max_v = float(self.get_parameter('max_v').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.max_cartesian_speed = float(
            self.get_parameter('max_cartesian_speed').value
        )

        self.own_robot_radius = float(
            self.get_parameter('own_robot_radius').value
        )
        self.other_robot_radius = float(
            self.get_parameter('other_robot_radius').value
        )
        self.puck_radius = float(self.get_parameter('puck_radius').value)
        self.stick_obstacle_radius = float(
            self.get_parameter('stick_obstacle_radius').value
        )
        self.minimum_clearance = float(
            self.get_parameter('minimum_clearance').value
        )
        self.emergency_clearance = float(
            self.get_parameter('emergency_clearance').value
        )
        self.obstacle_influence_clearance = float(
            self.get_parameter('obstacle_influence_clearance').value
        )
        self.obstacle_timeout = float(
            self.get_parameter('obstacle_timeout').value
        )
        self.include_pucks = bool(
            self.get_parameter('include_pucks_as_obstacles').value
        )

        self.clf_rate = float(self.get_parameter('clf_rate').value)
        self.cbf_rate = float(self.get_parameter('cbf_rate').value)
        self.clf_slack_weight = float(
            self.get_parameter('clf_slack_weight').value
        )
        self.qp_solver = str(
            self.get_parameter('qp_solver').value
        ).strip()

        self.stick_tip_offset_from_point = float(
            self.get_parameter('stick_tip_offset_from_point').value
        )
        self.precontact_gap = float(
            self.get_parameter('precontact_gap').value
        )
        self.robot4_tip_precontact_gap = float(
            self.get_parameter('robot4_tip_precontact_gap').value
        )
        self.robot4_pass_stage_tolerance = float(
            self.get_parameter('robot4_pass_stage_tolerance').value
        )
        self.robot4_precontact_advance_distance = float(
            self.get_parameter('robot4_precontact_advance_distance').value
        )
        self.robot4_precontact_speed = float(
            self.get_parameter('robot4_precontact_speed').value
        )
        self.robot4_precontact_timeout = float(
            self.get_parameter('robot4_precontact_timeout').value
        )
        self.robot4_manual_forward_distance = float(
            self.get_parameter('robot4_manual_forward_distance').value
        )
        self.robot4_manual_forward_speed = float(
            self.get_parameter('robot4_manual_forward_speed').value
        )
        self.robot4_manual_forward_timeout = float(
            self.get_parameter('robot4_manual_forward_timeout').value
        )
        self.robot4_manual_forward_completion_tolerance = float(
            self.get_parameter(
                'robot4_manual_forward_completion_tolerance'
            ).value
        )
        self.robot4_manual_forward_heading_gate = math.radians(
            float(
                self.get_parameter(
                    'robot4_manual_forward_heading_gate_deg'
                ).value
            )
        )
        self.robot4_precontact_heading_gate = math.radians(
            float(
                self.get_parameter(
                    'robot4_precontact_heading_gate_deg'
                ).value
            )
        )
        self.robot4_precontact_target_gap = float(
            self.get_parameter('robot4_precontact_target_gap').value
        )
        self.robot4_precontact_lateral_tolerance = float(
            self.get_parameter(
                'robot4_precontact_lateral_tolerance'
            ).value
        )
        self.robot4_precontact_slow_zone = float(
            self.get_parameter('robot4_precontact_slow_zone').value
        )
        self.pass_stroke_distance = float(
            self.get_parameter('pass_stroke_distance').value
        )
        self.shot_stroke_distance = float(
            self.get_parameter('shot_stroke_distance').value
        )
        self.contact_max_v = float(
            self.get_parameter('contact_max_v').value
        )
        self.contact_max_w = float(
            self.get_parameter('contact_max_w').value
        )
        self.contact_timeout = float(
            self.get_parameter('contact_timeout').value
        )

        self.robot4_stick_angle_offset = math.radians(
            float(
                self.get_parameter(
                    'robot4_stick_angle_offset_deg'
                ).value
            )
        )
        self.robot4_swing_direction = int(
            self.get_parameter('robot4_swing_direction').value
        )
        self.robot4_flip_stage_diameter = bool(
            self.get_parameter('robot4_flip_stage_diameter').value
        )
        self.robot4_backswing_angle = math.radians(
            float(
                self.get_parameter(
                    'robot4_backswing_angle_deg'
                ).value
            )
        )
        self.robot4_follow_through_angle = math.radians(
            float(
                self.get_parameter(
                    'robot4_follow_through_angle_deg'
                ).value
            )
        )
        self.robot4_backswing_max_w = float(
            self.get_parameter(
                'robot4_backswing_max_w'
            ).value
        )
        self.robot4_swing_start_w = float(
            self.get_parameter(
                'robot4_swing_start_w'
            ).value
        )
        self.robot4_swing_max_w = float(
            self.get_parameter(
                'robot4_swing_max_w'
            ).value
        )
        self.robot4_swing_angular_acceleration = float(
            self.get_parameter(
                'robot4_swing_angular_acceleration'
            ).value
        )
        self.robot4_preimpact_accel_fraction = float(
            self.get_parameter(
                'robot4_preimpact_accel_fraction'
            ).value
        )
        self.robot4_swing_timeout = float(
            self.get_parameter(
                'robot4_swing_timeout'
            ).value
        )
        self.robot4_extra_stage_clearance = float(
            self.get_parameter(
                'robot4_extra_stage_clearance'
            ).value
        )

        self.robot4_pass_stage_distance = float(
            self.get_parameter(
                'robot4_pass_stage_distance'
            ).value
        )
        
        self.robot4_swing_forward_speed = float(
            self.get_parameter(
                'robot4_swing_forward_speed'
            ).value
        )

        self.puck_receive_tolerance = float(
            self.get_parameter('puck_receive_tolerance').value
        )
        self.puck_receive_required_cycles = int(
            self.get_parameter('puck_receive_required_cycles').value
        )
        self.puck_motion_epsilon = float(
            self.get_parameter('puck_motion_epsilon').value
        )
        self.ignored_obstacle_topics = set(
            str(topic)
            for topic in self.get_parameter(
                'ignored_obstacle_topics'
            ).value
        )

        positive_values = {
            'point_offset': self.point_offset,
            'position_tolerance': self.position_tolerance,
            'max_v': self.max_v,
            'max_w': self.max_w,
            'max_cartesian_speed': self.max_cartesian_speed,
            'own_robot_radius': self.own_robot_radius,
            'other_robot_radius': self.other_robot_radius,
            'puck_radius': self.puck_radius,
            'minimum_clearance': self.minimum_clearance,
            'obstacle_timeout': self.obstacle_timeout,
            'clf_rate': self.clf_rate,
            'cbf_rate': self.cbf_rate,
            'clf_slack_weight': self.clf_slack_weight,
            'stick_tip_offset_from_point':
                self.stick_tip_offset_from_point,
            'pass_stroke_distance': self.pass_stroke_distance,
            'shot_stroke_distance': self.shot_stroke_distance,
            'contact_max_v': self.contact_max_v,
            'contact_max_w': self.contact_max_w,
            'contact_timeout': self.contact_timeout,
            'robot4_backswing_angle':
                self.robot4_backswing_angle,
            'robot4_follow_through_angle':
                self.robot4_follow_through_angle,
            'robot4_backswing_max_w':
                self.robot4_backswing_max_w,
            'robot4_swing_start_w':
                self.robot4_swing_start_w,
            'robot4_swing_max_w':
                self.robot4_swing_max_w,
            'robot4_swing_angular_acceleration':
                self.robot4_swing_angular_acceleration,
            'robot4_swing_timeout':
                self.robot4_swing_timeout,
            'receiver_departure_heading_tolerance':
                self.receiver_departure_heading_tolerance,
            'receiver_departure_max_w':
                self.receiver_departure_max_w,
            'robot4_departure_heading_tolerance':
                self.robot4_departure_heading_tolerance,
            'robot4_departure_max_w':
                self.robot4_departure_max_w,
            'puck_receive_tolerance': self.puck_receive_tolerance,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f'{name} must be positive.')

        if self.precontact_gap < 0.0:
            raise ValueError('precontact_gap cannot be negative.')
        if self.robot4_tip_precontact_gap < 0.0:
            raise ValueError(
                'robot4_tip_precontact_gap cannot be negative.'
            )
        if self.robot4_pass_stage_tolerance <= 0.0:
            raise ValueError(
                'robot4_pass_stage_tolerance must be positive.'
            )
        if self.robot4_precontact_advance_distance < 0.0:
            raise ValueError(
                'robot4_precontact_advance_distance cannot be negative.'
            )
        if self.robot4_precontact_speed <= 0.0:
            raise ValueError('robot4_precontact_speed must be positive.')
        if self.robot4_precontact_timeout <= 0.0:
            raise ValueError('robot4_precontact_timeout must be positive.')
        if self.robot4_manual_forward_distance < 0.0:
            raise ValueError(
                'robot4_manual_forward_distance cannot be negative.'
            )
        if self.robot4_manual_forward_speed <= 0.0:
            raise ValueError(
                'robot4_manual_forward_speed must be positive.'
            )
        if self.robot4_manual_forward_timeout <= 0.0:
            raise ValueError(
                'robot4_manual_forward_timeout must be positive.'
            )
        if not (
            0.0
            <= self.robot4_manual_forward_completion_tolerance
            < self.robot4_manual_forward_distance
        ):
            raise ValueError(
                'robot4_manual_forward_completion_tolerance must be '
                'nonnegative and smaller than the requested distance.'
            )
        if self.robot4_manual_forward_heading_gate <= 0.0:
            raise ValueError(
                'robot4_manual_forward_heading_gate_deg must be positive.'
            )
        if self.robot4_precontact_heading_gate <= 0.0:
            raise ValueError(
                'robot4_precontact_heading_gate_deg must be positive.'
            )
        if self.robot4_precontact_lateral_tolerance <= 0.0:
            raise ValueError(
                'robot4_precontact_lateral_tolerance must be positive.'
            )
        if self.robot4_precontact_slow_zone <= 0.0:
            raise ValueError('robot4_precontact_slow_zone must be positive.')
        if self.robot4_extra_stage_clearance < 0.0:
            raise ValueError(
                'robot4_extra_stage_clearance cannot be negative.'
            )
        if self.robot4_swing_forward_speed < 0.0:
            raise ValueError(
                'robot4_swing_forward_speed cannot be negative.'
            )
        if self.robot4_swing_direction not in (-1, 1):
            raise ValueError(
                'robot4_swing_direction must be +1 or -1.'
            )
        if not (0.0 < self.robot4_preimpact_accel_fraction <= 1.0):
            raise ValueError(
                'robot4_preimpact_accel_fraction must be in (0, 1].'
            )
        if (
            self.robot4_swing_start_w
            > self.robot4_swing_max_w
        ):
            raise ValueError(
                'robot4_swing_start_w cannot exceed '
                'robot4_swing_max_w.'
            )
        if (
            self.robot4_backswing_angle
            + self.robot4_follow_through_angle
            >= math.pi
        ):
            raise ValueError(
                'Robot 4 total swing angle must be less '
                'than 180 degrees.'
            )
        if self.puck_receive_required_cycles < 1:
            raise ValueError(
                'puck_receive_required_cycles must be at least 1.'
            )
        if not self.qp_solver:
            raise ValueError('qp_solver cannot be empty.')

        # Desired distance from the controlled point to the target puck
        # center at the staging pose.
        self.precontact_distance = (
            self.stick_tip_offset_from_point
            + self.puck_radius
            + self.precontact_gap
        )

        # Protect the stick tip and puck during normal navigation. This
        # boundary excludes the desired pre-contact gap, so the staging
        # target remains outside the CBF boundary instead of lying on it.
        self.target_puck_safe_distance = (
            self.stick_tip_offset_from_point
            + self.puck_radius
        )
        self.robot4_stage_puck_safe_distance = (
            self.own_robot_radius
            + self.puck_radius
            + 0.03
        )

        # ============================================================
        # Runtime state
        # ============================================================
        self.puck_x = 0.0
        self.puck_y = 0.0
        self.puck_pose_received = False
        self.previous_puck_position: Optional[Tuple[float, float]] = None

        self.receive_x = 0.0
        self.receive_y = 0.0
        self.receive_position_frozen = False
        self.receiver_departure_aligned = False
        self.robot4_departure_aligned = False

        # Freeze Robot 4's selected pass-stage point once Robot 5 has
        # reached its receiving pose. This prevents mocap noise from moving
        # the navigation target while Robot 4 is approaching it.
        self.robot4_stage_target: Optional[Tuple[float, float]] = None

        self.state = self.WAIT_FOR_POSES
        self.state_start_ns = self.get_clock().now().nanoseconds
        self.handoff_complete = False

        self.puck_receive_counter = 0
        self.contact_start_point: Optional[Tuple[float, float]] = None
        self.contact_target_point: Optional[Tuple[float, float]] = None

        # Robot-4 stick-tip pass runtime values. The pass controller tracks
        # the physical stick tip rather than the chassis controlled point.
        self.robot4_pass_heading: Optional[float] = None
        self.robot4_tip_start: Optional[Tuple[float, float]] = None
        self.robot4_tip_target: Optional[Tuple[float, float]] = None
        self.robot4_pass_puck_start: Optional[Tuple[float, float]] = None
        self.robot4_pass_required_progress = 0.0
        self.robot4_precontact_start: Optional[Tuple[float, float]] = None
        self.robot4_precontact_heading: Optional[float] = None
        self.robot4_precontact_puck_start: Optional[Tuple[float, float]] = None

        # Manual-forward runtime values. Alignment and translation timing are
        # deliberately separated so time spent correcting the heading does not
        # consume the requested forward-motion time.
        self.robot4_manual_forward_heading: Optional[float] = None
        self.robot4_manual_forward_start: Optional[Tuple[float, float]] = None
        self.robot4_manual_forward_puck_start: Optional[Tuple[float, float]] = None
        self.robot4_manual_translation_started = False
        self.robot4_manual_translation_start_ns: Optional[int] = None
        self.robot4_manual_alignment_start_ns: Optional[int] = None
        self.robot4_manual_forward_failed = False

        # Robot 4 rotational-pass runtime values.
        self.robot4_preload_start_heading: Optional[float] = None
        self.robot4_swing_start_heading: Optional[float] = None
        self.robot4_impact_heading: Optional[float] = None
        self.robot4_swing_total_angle = (
            self.robot4_backswing_angle
            + self.robot4_follow_through_angle
        )

        self.obstacle_poses: Dict[str, ObstacleState] = {}
        self.obstacle_subscriptions = []
        self.dynamic_obstacle_subscriptions = []
        self.subscribed_obstacle_topics = set()

        self.last_waiting_log_ns = 0
        self.last_debug_log_ns = 0
        self.last_safety_log_ns = 0
        self.last_qp_log_ns = 0

        # ============================================================
        # ROS interfaces
        # ============================================================
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.robot4_cmd_pub = self.create_publisher(
            Twist,
            f'/robot{self.robot4.robot_id}/cmd_vel',
            10,
        )
        self.robot5_cmd_pub = self.create_publisher(
            Twist,
            f'/robot{self.robot5.robot_id}/cmd_vel',
            10,
        )

        self.robot4_pose_sub = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{self.robot4.robot_id}/pose',
            lambda msg: self.robot_pose_callback(msg, self.robot4),
            self.best_effort_qos,
        )
        self.robot5_pose_sub = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{self.robot5.robot_id}/pose',
            lambda msg: self.robot_pose_callback(msg, self.robot5),
            self.best_effort_qos,
        )
        self.puck_pose_sub = self.create_subscription(
            PoseStamped,
            self.puck_topic,
            self.puck_pose_callback,
            self.best_effort_qos,
        )

        # Other robots are statically known obstacle topics.
        for robot_id in range(1, 11):
            if robot_id in {
                self.robot4.robot_id,
                self.robot5.robot_id,
            }:
                continue

            topic = f'/vrpn_mocap/dji_robot_{robot_id}/pose'
            subscription = self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, topic_name=topic:
                    self.obstacle_pose_callback(
                        msg,
                        topic_name,
                        self.other_robot_radius,
                    ),
                self.best_effort_qos,
            )
            self.obstacle_subscriptions.append(subscription)
            self.subscribed_obstacle_topics.add(topic)

        self.discovery_timer = self.create_timer(
            0.50,
            self.discover_object_obstacle_topics,
        )
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            f'T4 coordinator: Robot {self.robot4.robot_id} passes to '
            f'Robot {self.robot5.robot_id}, then Robot '
            f'{self.robot5.robot_id} shoots.'
        )
        self.get_logger().info(
            f'Green puck topic: {self.puck_topic}'
        )
        self.get_logger().info(
            f'Goal: ({self.goal_x:.3f}, {self.goal_y:.3f})'
        )
        self.get_logger().info(
            f'Pre-contact controlled-point distance: '
            f'{self.precontact_distance:.3f} m'
        )
        self.get_logger().info(
            f'Target-puck CBF distance: '
            f'{self.target_puck_safe_distance:.3f} m'
        )
        self.get_logger().info(
            f'Robot 4 opposite-diameter staging: '
            f'{self.robot4_flip_stage_diameter}'
        )
        self.get_logger().info(
            f'Robot 4 extra stage clearance: '
            f'{self.robot4_extra_stage_clearance:.3f} m; '
            f'swing forward speed: '
            f'{self.robot4_swing_forward_speed:.3f} m/s'
        )

    # ================================================================
    # Callbacks and topic discovery
    # ================================================================
    def robot_pose_callback(
        self,
        msg: PoseStamped,
        robot: RobotState,
    ) -> None:
        robot.x = msg.pose.position.x
        robot.y = msg.pose.position.y
        robot.theta = yaw_from_quaternion(msg.pose.orientation)
        robot.pose_received = True

    def puck_pose_callback(self, msg: PoseStamped) -> None:
        self.previous_puck_position = (
            (self.puck_x, self.puck_y)
            if self.puck_pose_received
            else None
        )
        self.puck_x = msg.pose.position.x
        self.puck_y = msg.pose.position.y
        self.puck_pose_received = True

    def obstacle_pose_callback(
        self,
        msg: PoseStamped,
        topic_name: str,
        obstacle_radius: float,
    ) -> None:
        self.obstacle_poses[topic_name] = ObstacleState(
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            radius=obstacle_radius,
            update_time_ns=self.get_clock().now().nanoseconds,
            topic=topic_name,
        )

    def discover_object_obstacle_topics(self) -> None:
        pattern = re.compile(
            r'^/vrpn_mocap/'
            r'(hockey_puck_[^/]+|hockey_sticks?_[^/]+)'
            r'/pose$'
        )

        for topic_name, topic_types in self.get_topic_names_and_types():
            if (
                'geometry_msgs/msg/PoseStamped'
                not in topic_types
            ):
                continue
            if topic_name in self.subscribed_obstacle_topics:
                continue
            if topic_name == self.puck_topic:
                continue
            if topic_name in self.ignored_obstacle_topics:
                continue

            match = pattern.match(topic_name)
            if match is None:
                continue

            object_name = match.group(1)
            if object_name.startswith('hockey_puck_'):
                if not self.include_pucks:
                    continue
                radius = self.puck_radius
            else:
                radius = self.stick_obstacle_radius

            subscription = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg,
                       discovered_topic=topic_name,
                       obstacle_radius=radius:
                    self.obstacle_pose_callback(
                        msg,
                        discovered_topic,
                        obstacle_radius,
                    ),
                self.best_effort_qos,
            )
            self.dynamic_obstacle_subscriptions.append(subscription)
            self.subscribed_obstacle_topics.add(topic_name)
            self.get_logger().info(
                f'Discovered obstacle topic: {topic_name}'
            )

    # ================================================================
    # Geometry
    # ================================================================
    def controlled_point(
        self,
        robot: RobotState,
    ) -> Tuple[float, float]:
        return (
            robot.x + self.point_offset * math.cos(robot.theta),
            robot.y + self.point_offset * math.sin(robot.theta),
        )

    def calculate_receiver_position(self) -> Tuple[float, float]:
        """Fixed receive point to the right side of the goal."""
        dx = self.goal_x - self.puck_x
        dy = self.goal_y - self.puck_y
        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            raise ValueError(
                'Goal and puck positions cannot be equal.'
            )

        direction_x = dx / distance
        direction_y = dy / distance

        # Right-hand normal of puck-to-goal direction.
        right_x = direction_y
        right_y = -direction_x

        receive_x = (
            self.goal_x
            - self.receiver_backoff * direction_x
            + self.receiver_right_offset * right_x
        )
        receive_y = (
            self.goal_y
            - self.receiver_backoff * direction_y
            + self.receiver_right_offset * right_y
        )
        return receive_x, receive_y

    def heading_between(
        self,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
    ) -> float:
        return math.atan2(target_y - source_y, target_x - source_x)

    def live_receiver_position(self) -> Tuple[float, float]:
        """
        Return the selected receiver robot's latest VRPN position.

        receiver_robot_id determines the subscribed topic, so changing the
        receiver ID automatically changes the live target used by Robot 4.
        """
        if not self.robot5.pose_received:
            raise RuntimeError(
                'The receiving robot VRPN pose has not been received.'
            )

        return self.robot5.x, self.robot5.y

    def stage_point_behind_puck(
        self,
        target_x: float,
        target_y: float,
    ) -> Tuple[float, float, float]:
        heading = self.heading_between(
            self.puck_x,
            self.puck_y,
            target_x,
            target_y,
        )
        stage_x = (
            self.puck_x
            - self.precontact_distance * math.cos(heading)
        )
        stage_y = (
            self.puck_y
            - self.precontact_distance * math.sin(heading)
        )
        return stage_x, stage_y, heading

    # ================================================================
    # Safety and CLF-CBF-QP
    # ================================================================
    def target_puck_contact_enabled(self) -> bool:
        return self.state in {
            self.PASS_WITH_R4,
            self.SHOOT_WITH_R5,
        }
        

    def collect_active_obstacles(
        self,
        controlled_robot: RobotState,
        px: float,
        py: float,
    ) -> Tuple[
        List[Tuple[float, float, float]],
        bool,
    ]:
        """
        Return active obstacles as (x, y, safe_distance).

        For normal obstacles, safe_distance follows the T1/T3 conservative
        chassis envelope. For the target puck during staging, safe_distance
        is based on the attached stick-tip reach. During deliberate contact
        strokes, the target puck constraint is disabled.
        """
        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(self.obstacle_timeout * 1e9)
        active: List[Tuple[float, float, float]] = []
        emergency_stop = False

        def consider(
            obstacle_x: float,
            obstacle_y: float,
            safe_distance: float,
            obstacle_name: str,
            emergency_threshold: Optional[float] = None,
        ) -> None:
            nonlocal emergency_stop

            distance = math.hypot(
                px - obstacle_x,
                py - obstacle_y,
            )
            clearance = distance - safe_distance

            threshold = (
                self.emergency_clearance
                if emergency_threshold is None
                else emergency_threshold
            )

            if clearance <= threshold:
                emergency_stop = True

                self.log_periodically(
                    f'Emergency obstacle for Robot '
                    f'{controlled_robot.robot_id}: '
                    f'{obstacle_name}; '
                    f'controlled_point=({px:.3f},{py:.3f}); '
                    f'obstacle=({obstacle_x:.3f},{obstacle_y:.3f}); '
                    f'distance={distance:.3f} m; '
                    f'safe_distance={safe_distance:.3f} m; '
                    f'clearance={clearance:.3f} m; '
                    f'emergency_limit={threshold:.3f} m',
                    'last_safety_log_ns',
                    1.0,
                )

            if clearance < self.obstacle_influence_clearance:
                active.append((
                    obstacle_x,
                    obstacle_y,
                    safe_distance,
                ))

        # Other coordinated robot.
        other_robot = (
            self.robot5
            if controlled_robot.robot_id == self.robot4.robot_id
            else self.robot4
        )
        if other_robot.pose_received:
            consider(
                other_robot.x,
                other_robot.y,
                self.own_robot_radius
                + self.other_robot_radius
                + self.point_offset
                + self.minimum_clearance,
                f'coordinated_robot_{other_robot.robot_id}',
            )

        # Other tracked obstacles.
        for obstacle in self.obstacle_poses.values():
            if now_ns - obstacle.update_time_ns > timeout_ns:
                continue
            if obstacle.topic in self.ignored_obstacle_topics:
                continue

            consider(
                obstacle.x,
                obstacle.y,
                self.own_robot_radius
                + obstacle.radius
                + self.point_offset
                + self.minimum_clearance,
                obstacle.topic,
            )

        # Selected green puck: protected in staging/navigation, deliberately
        # contactable only in pass/shoot states.
        target_puck_safe_distance = self.target_puck_safe_distance
 
        if (
            controlled_robot.robot_id == self.robot4.robot_id
            and self.state in {
                self.MOVE_R4_TO_PASS_STAGE,
                self.ALIGN_R4_TO_PASS,
            }
        ):
            target_puck_safe_distance = (
                self.robot4_stage_puck_safe_distance
            )
        
            consider(
                self.puck_x,
                self.puck_y,
                target_puck_safe_distance,
                'target_green_puck',
                emergency_threshold=0.0,
            )

        return active, emergency_stop

    def solve_clf_cbf_qp(
        self,
        robot: RobotState,
        px: float,
        py: float,
        target_x: float,
        target_y: float,
        nominal_ux: float,
        nominal_uy: float,
        max_v: float,
        max_w: float,
    ) -> Tuple[float, float, bool, bool]:
        obstacles, emergency_stop = self.collect_active_obstacles(
            robot,
            px,
            py,
        )
        if emergency_stop:
            return 0.0, 0.0, False, True

        # z = [ux, uy, delta]
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

        # CLF: e^T u - delta <= -c V
        ex = px - target_x
        ey = py - target_y
        lyapunov_value = 0.5 * (ex * ex + ey * ey)

        g_rows.append([ex, ey, -1.0])
        h_values.append(-self.clf_rate * lyapunov_value)

        # delta >= 0
        g_rows.append([0.0, 0.0, -1.0])
        h_values.append(0.0)

        # CBF: h_dot + alpha h >= 0
        for obstacle_x, obstacle_y, safe_distance in obstacles:
            dx = px - obstacle_x
            dy = py - obstacle_y
            barrier_value = (
                dx * dx
                + dy * dy
                - safe_distance * safe_distance
            )

            g_rows.append([
                -2.0 * dx,
                -2.0 * dy,
                0.0,
            ])
            h_values.append(self.cbf_rate * barrier_value)

        cos_theta = math.cos(robot.theta)
        sin_theta = math.sin(robot.theta)

        # Linear speed constraints.
        g_rows.extend([
            [cos_theta, sin_theta, 0.0],
            [-cos_theta, -sin_theta, 0.0],
        ])
        h_values.extend([max_v, max_v])

        # Angular speed constraints.
        omega_ux = -sin_theta / self.point_offset
        omega_uy = cos_theta / self.point_offset
        g_rows.extend([
            [omega_ux, omega_uy, 0.0],
            [-omega_ux, -omega_uy, 0.0],
        ])
        h_values.extend([max_w, max_w])

        # Cartesian component bounds.
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

        solver_candidates = []
        for candidate in (
            self.qp_solver,
            'quadprog',
            'cvxopt',
            'osqp',
        ):
            if candidate not in solver_candidates:
                solver_candidates.append(candidate)

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

        speed = math.hypot(ux, uy)
        if speed > self.max_cartesian_speed:
            scale = self.max_cartesian_speed / speed
            ux *= scale
            uy *= scale

        return ux, uy, True, False

    # ================================================================
    # Command helpers
    # ================================================================
    def publisher_for(self, robot: RobotState):
        if robot.robot_id == self.robot4.robot_id:
            return self.robot4_cmd_pub
        return self.robot5_cmd_pub

    def publish_stop(self, robot: RobotState) -> None:
        self.publisher_for(robot).publish(Twist())

    def stop_both(self) -> None:
        self.publish_stop(self.robot4)
        self.publish_stop(self.robot5)

    def navigate_robot_to(
        self,
        robot: RobotState,
        target_x: float,
        target_y: float,
        max_v: Optional[float] = None,
        max_w: Optional[float] = None,
    ) -> Tuple[bool, float]:
        px, py = self.controlled_point(robot)
        error_x = target_x - px
        error_y = target_y - py
        position_error = math.hypot(error_x, error_y)

        nominal_ux = self.k_att * error_x
        nominal_uy = self.k_att * error_y

        ux, uy, success, emergency = self.solve_clf_cbf_qp(
            robot=robot,
            px=px,
            py=py,
            target_x=target_x,
            target_y=target_y,
            nominal_ux=nominal_ux,
            nominal_uy=nominal_uy,
            max_v=self.max_v if max_v is None else max_v,
            max_w=self.max_w if max_w is None else max_w,
        )

        if emergency:
            self.publish_stop(robot)
            return False, position_error

        if not success:
            self.publish_stop(robot)
            self.log_periodically(
                'CLF-CBF-QP failed to return a feasible command.',
                'last_qp_log_ns',
                1.0,
            )
            return False, position_error

        v = (
            math.cos(robot.theta) * ux
            + math.sin(robot.theta) * uy
        )
        omega = (
            -math.sin(robot.theta) * ux
            + math.cos(robot.theta) * uy
        ) / self.point_offset

        v_limit = self.max_v if max_v is None else max_v
        w_limit = self.max_w if max_w is None else max_w

        cmd = Twist()
        cmd.linear.x = clamp(v, -v_limit, v_limit)
        cmd.angular.z = clamp(omega, -w_limit, w_limit)
        self.publisher_for(robot).publish(cmd)
        return True, position_error

    def align_robot(
        self,
        robot: RobotState,
        desired_heading: float,
        max_w: Optional[float] = None,
    ) -> Tuple[bool, float]:
        error = wrap_angle(desired_heading - robot.theta)
        if abs(error) <= self.heading_tolerance:
            self.publish_stop(robot)
            return True, error

        cmd = Twist()
        w_limit = self.max_w if max_w is None else max_w
        cmd.angular.z = clamp(
            self.k_heading * error,
            -w_limit,
            w_limit,
        )
        self.publisher_for(robot).publish(cmd)
        return False, error

    def begin_contact_stroke(
        self,
        robot: RobotState,
        heading: float,
        stroke_distance: float,
    ) -> None:
        px, py = self.controlled_point(robot)
        self.contact_start_point = (px, py)
        self.contact_target_point = (
            px + stroke_distance * math.cos(heading),
            py + stroke_distance * math.sin(heading),
        )
        self.state_start_ns = self.get_clock().now().nanoseconds

    def execute_contact_stroke(
        self,
        robot: RobotState,
    ) -> Tuple[bool, float]:
        if self.contact_target_point is None:
            raise RuntimeError('Contact target was not initialized.')

        target_x, target_y = self.contact_target_point
        success, error = self.navigate_robot_to(
            robot,
            target_x,
            target_y,
            max_v=self.contact_max_v,
            max_w=self.contact_max_w,
        )

        elapsed = (
            self.get_clock().now().nanoseconds
            - self.state_start_ns
        ) / 1e9

        finished = (
            error <= self.position_tolerance
            or elapsed >= self.contact_timeout
        )
        if finished:
            self.publish_stop(robot)

        return finished and success, error

    def robot4_stick_tip_position(self) -> Tuple[float, float]:
        """Return the world position of Robot 4's physical stick tip."""
        theta = self.robot4.theta
        stick_heading = theta + self.robot4_stick_angle_offset
        return (
            self.robot4.x
            + self.point_offset * math.cos(theta)
            + self.stick_tip_offset_from_point * math.cos(stick_heading),
            self.robot4.y
            + self.point_offset * math.sin(theta)
            + self.stick_tip_offset_from_point * math.sin(stick_heading),
        )

    def robot4_stick_tip_jacobian(self) -> np.ndarray:
        """Map Robot 4 [v, omega] to physical stick-tip velocity."""
        theta = self.robot4.theta
        stick_heading = theta + self.robot4_stick_angle_offset

        radius_x = (
            self.point_offset * math.cos(theta)
            + self.stick_tip_offset_from_point * math.cos(stick_heading)
        )
        radius_y = (
            self.point_offset * math.sin(theta)
            + self.stick_tip_offset_from_point * math.sin(stick_heading)
        )

        return np.array([
            [math.cos(theta), -radius_y],
            [math.sin(theta), radius_x],
        ], dtype=float)


    def begin_robot4_precontact_adjustment(self) -> None:
        """Freeze the shot line and initialize a measured straight creep."""
        self.robot4_precontact_heading = self.robot4_shot_heading()

        # Measure the physical forward adjustment from Robot 4's live VRPN
        # chassis pose. 
        self.robot4_precontact_start = (self.robot4.x, self.robot4.y)
        self.robot4_precontact_puck_start = (self.puck_x, self.puck_y)
        self.state_start_ns = self.get_clock().now().nanoseconds

        self.get_logger().info(
            f'Robot 4 measured pre-contact initialized: '
            f'advance={self.robot4_precontact_advance_distance:.3f} m; '
            f'speed={self.robot4_precontact_speed:.3f} m/s; '
            f'heading={math.degrees(self.robot4_precontact_heading):.1f} deg.'
        )

    def execute_robot4_precontact_adjustment(
        self,
    ) -> Tuple[bool, float, float, float]:
        """Advance Robot 4 straight toward the puck using VRPN distance.

        The heading is frozen at the puck-to-Robot-5 direction. Translation
        pauses whenever the heading error is too large. The phase stops as
        soon as the puck moves toward Robot 5, or after the requested chassis
        advance/timeout. 
        """
        if (
            self.robot4_precontact_start is None
            or self.robot4_precontact_heading is None
            or self.robot4_precontact_puck_start is None
        ):
            raise RuntimeError(
                'Robot 4 pre-contact adjustment was not initialized.'
            )

        shot_ux = math.cos(self.robot4_precontact_heading)
        shot_uy = math.sin(self.robot4_precontact_heading)
        normal_x = -shot_uy
        normal_y = shot_ux

        start_x, start_y = self.robot4_precontact_start
        progress = (
            (self.robot4.x - start_x) * shot_ux
            + (self.robot4.y - start_y) * shot_uy
        )
        remaining_advance = max(
            0.0,
            self.robot4_precontact_advance_distance - progress,
        )


        tip_x, tip_y = self.robot4_stick_tip_position()
        tip_to_puck_x = self.puck_x - tip_x
        tip_to_puck_y = self.puck_y - tip_y
        longitudinal_gap = (
            tip_to_puck_x * shot_ux
            + tip_to_puck_y * shot_uy
            - self.puck_radius
        )
        lateral_error = (
            tip_to_puck_x * normal_x
            + tip_to_puck_y * normal_y
        )

        px, py = self.controlled_point(self.robot4)
        _, emergency = self.collect_active_obstacles(self.robot4, px, py)
        if emergency:
            self.publish_stop(self.robot4)
            return False, longitudinal_gap, lateral_error, remaining_advance

        heading_error = wrap_angle(
            self.robot4_precontact_heading - self.robot4.theta
        )

        abs_heading_error = abs(heading_error)
        soft_heading_gate = 0.5 * self.robot4_precontact_heading_gate

        if abs_heading_error > self.robot4_precontact_heading_gate:
            linear_speed = 0.0
        elif remaining_advance <= 0.0:
            linear_speed = 0.0
        else:
            heading_scale = 1.0
            if abs_heading_error > soft_heading_gate:
                heading_scale = max(
                    0.35,
                    (
                        self.robot4_precontact_heading_gate
                        - abs_heading_error
                    ) / max(soft_heading_gate, 1e-6),
                )

            distance_scale = 1.0
            if remaining_advance < 0.03:
                distance_scale = max(0.35, remaining_advance / 0.03)

            linear_speed = (
                self.robot4_precontact_speed
                * heading_scale
                * distance_scale
            )

        cmd = Twist()
        cmd.linear.x = clamp(
            linear_speed, 0.0, self.robot4_precontact_speed
        )
        cmd.angular.z = clamp(
            self.k_heading * heading_error,
            -self.contact_max_w,
            self.contact_max_w,
        )
        self.robot4_cmd_pub.publish(cmd)

        elapsed = (
            self.get_clock().now().nanoseconds - self.state_start_ns
        ) / 1e9
        puck_progress = (
            (self.puck_x - self.robot4_precontact_puck_start[0]) * shot_ux
            + (self.puck_y - self.robot4_precontact_puck_start[1]) * shot_uy
        )
        puck_touched = puck_progress >= 0.5 * self.puck_motion_epsilon

        finished = (
            puck_touched
            or progress >= self.robot4_precontact_advance_distance
            or elapsed >= self.robot4_precontact_timeout
        )

        if finished:
            self.publish_stop(self.robot4)
            if puck_touched:
                self.get_logger().info(
                    'Robot 4 detected puck motion during pre-contact; '
                    'starting the shooting follow-through.'
                )
            elif elapsed >= self.robot4_precontact_timeout:
                self.get_logger().warning(
                    'Robot 4 pre-contact timed out before the full requested '
                    f'advance; progress={progress:.3f} m.'
                )

        return (
            finished,
            longitudinal_gap,
            lateral_error,
            remaining_advance,
        )


    def begin_robot4_manual_forward(self) -> None:
        """Initialize one explicit final forward nudge before the shot.

        Robot 4 has already reached the experimentally useful shooting pose
        when this state begins.  Freeze the current chassis heading instead of
        replacing it with the puck-to-Robot-5 line.  The stick is mounted at an
        angle, so forcing the chassis onto the puck-to-receiver heading rotates
        the stick away from its good contact pose and keeps linear speed zero
        behind the heading gate.
        """
        self.robot4_manual_forward_heading = self.robot4.theta
        self.robot4_manual_forward_start = None
        self.robot4_manual_forward_puck_start = None
        self.robot4_manual_translation_started = False
        self.robot4_manual_translation_start_ns = None
        self.robot4_manual_alignment_start_ns = (
            self.get_clock().now().nanoseconds
        )
        self.robot4_manual_forward_failed = False
        self.state_start_ns = self.robot4_manual_alignment_start_ns

        nominal_translation_time = (
            self.robot4_manual_forward_distance
            / self.robot4_manual_forward_speed
        )
        # Account for the intentional near-target slowdown and heading corrections.
        effective_translation_timeout = max(
            self.robot4_manual_forward_timeout,
            2.0 * nominal_translation_time + 2.0,
        )

        self.get_logger().info(
            f'Robot 4 manual forward nudge initialized: '
            f'distance={self.robot4_manual_forward_distance:.3f} m; '
            f'speed={self.robot4_manual_forward_speed:.3f} m/s; '
            f'frozen_current_heading='
            f'{math.degrees(self.robot4_manual_forward_heading):.1f} deg; '
            f'translation_timeout={effective_translation_timeout:.2f} s; '
            f'completion_tolerance='
            f'{self.robot4_manual_forward_completion_tolerance:.3f} m.'
        )

    def execute_robot4_manual_forward(
        self,
    ) -> Tuple[bool, bool, float, float]:
        """Align first, then move the requested measured forward distance.

        Returns (finished, succeeded, progress, remaining). 
        """
        if (
            self.robot4_manual_forward_heading is None
            or self.robot4_manual_alignment_start_ns is None
        ):
            raise RuntimeError(
                'Robot 4 manual forward nudge was not initialized.'
            )

        if self.robot4_manual_forward_failed:
            self.publish_stop(self.robot4)
            remaining = self.robot4_manual_forward_distance
            return True, False, 0.0, remaining

        now_ns = self.get_clock().now().nanoseconds
        shot_ux = math.cos(self.robot4_manual_forward_heading)
        shot_uy = math.sin(self.robot4_manual_forward_heading)
        heading_error = wrap_angle(
            self.robot4_manual_forward_heading - self.robot4.theta
        )

        px, py = self.controlled_point(self.robot4)
        _, emergency = self.collect_active_obstacles(
            self.robot4,
            px,
            py,
        )
        if emergency:
            self.publish_stop(self.robot4)
            return False, False, 0.0, self.robot4_manual_forward_distance

        # Phase 1: heading alignment. The forward-distance origin and movement
        # timer are not started until the heading is acceptable.
        if not self.robot4_manual_translation_started:
            if abs(heading_error) > self.robot4_manual_forward_heading_gate:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = clamp(
                    self.k_heading * heading_error,
                    -self.contact_max_w,
                    self.contact_max_w,
                )
                self.robot4_cmd_pub.publish(cmd)

                alignment_elapsed = (
                    now_ns - self.robot4_manual_alignment_start_ns
                ) / 1e9
                if alignment_elapsed >= self.robot4_manual_forward_timeout:
                    self.publish_stop(self.robot4)
                    self.robot4_manual_forward_failed = True
                    self.get_logger().error(
                        'Robot 4 manual nudge could not establish the '
                        'required heading; the shot is blocked.'
                    )
                    return (
                        True,
                        False,
                        0.0,
                        self.robot4_manual_forward_distance,
                    )
                return (
                    False,
                    False,
                    0.0,
                    self.robot4_manual_forward_distance,
                )

            self.publish_stop(self.robot4)
            self.robot4_manual_translation_started = True
            self.robot4_manual_forward_start = (
                self.robot4.x,
                self.robot4.y,
            )
            self.robot4_manual_forward_puck_start = (
                self.puck_x,
                self.puck_y,
            )
            self.robot4_manual_translation_start_ns = now_ns
            self.get_logger().info(
                'Robot 4 manual nudge heading accepted; starting measured '
                'forward translation now.'
            )

        if (
            self.robot4_manual_forward_start is None
            or self.robot4_manual_forward_puck_start is None
            or self.robot4_manual_translation_start_ns is None
        ):
            raise RuntimeError(
                'Robot 4 manual translation origin was not initialized.'
            )

        # Phase 2: measured translation.
        start_x, start_y = self.robot4_manual_forward_start
        progress = max(
            0.0,
            (self.robot4.x - start_x) * shot_ux
            + (self.robot4.y - start_y) * shot_uy,
        )
        remaining = max(
            0.0,
            self.robot4_manual_forward_distance - progress,
        )

        # Preserve heading during translation. Pause only for a genuinely large
        # error, small errors are corrected while continuing forward.
        hard_heading_gate = max(
            math.radians(25.0),
            2.0 * self.robot4_manual_forward_heading_gate,
        )
        if abs(heading_error) > hard_heading_gate:
            # Stop only if the robot has genuinely lost the frozen approach direction. 
            linear_speed = 0.0
        elif remaining <= 0.0:
            linear_speed = 0.0
        else:
            distance_scale = 1.0
            if remaining < 0.02:
                distance_scale = max(0.30, remaining / 0.02)
            heading_scale = max(
                0.35,
                1.0 - abs(heading_error) / hard_heading_gate,
            )
            linear_speed = (
                self.robot4_manual_forward_speed
                * distance_scale
                * heading_scale
            )

        cmd = Twist()
        cmd.linear.x = clamp(
            linear_speed,
            0.0,
            self.robot4_manual_forward_speed,
        )
        cmd.angular.z = clamp(
            self.k_heading * heading_error,
            -self.contact_max_w,
            self.contact_max_w,
        )
        self.robot4_cmd_pub.publish(cmd)

        puck_progress = (
            (self.puck_x - self.robot4_manual_forward_puck_start[0])
            * shot_ux
            + (self.puck_y - self.robot4_manual_forward_puck_start[1])
            * shot_uy
        )
        puck_moved = (
            puck_progress >= 0.5 * self.puck_motion_epsilon
        )

        translation_elapsed = (
            now_ns - self.robot4_manual_translation_start_ns
        ) / 1e9
        nominal_translation_time = (
            self.robot4_manual_forward_distance
            / self.robot4_manual_forward_speed
        )
        # Account for the intentional near-target slowdown and heading corrections.
        effective_translation_timeout = max(
            self.robot4_manual_forward_timeout,
            2.0 * nominal_translation_time + 2.0,
        )

        distance_completed = (
            remaining
            <= self.robot4_manual_forward_completion_tolerance
        )
        succeeded = puck_moved or distance_completed
        timed_out = (
            translation_elapsed >= effective_translation_timeout
            and not succeeded
        )
        finished = succeeded or timed_out

        if finished:
            self.publish_stop(self.robot4)
            if puck_moved:
                self.get_logger().info(
                    'Puck motion detected during Robot 4 manual nudge.'
                )
            elif succeeded:
                self.get_logger().info(
                    'Robot 4 completed the manual forward nudge: '
                    f'progress={progress:.3f} m; '
                    f'remaining={remaining:.3f} m; '
                    f'tolerance='
                    f'{self.robot4_manual_forward_completion_tolerance:.3f} m.'
                )
            else:
                self.robot4_manual_forward_failed = True
                self.get_logger().error(
                    'Robot 4 manual forward translation timed out before '
                    f'completion; progress={progress:.3f} m, '
                    f'requested={self.robot4_manual_forward_distance:.3f} m. '
                    'The shot is blocked.'
                )

        return finished, succeeded, progress, remaining

    def begin_robot4_tip_pass(self) -> None:
        """Freeze the shot line and initialize one short straight stroke."""
        self.robot4_pass_heading = self.robot4_shot_heading()
        tip_x, tip_y = self.robot4_stick_tip_position()
        self.robot4_tip_start = (tip_x, tip_y)
        self.robot4_pass_puck_start = (self.puck_x, self.puck_y)

        shot_ux = math.cos(self.robot4_pass_heading)
        shot_uy = math.sin(self.robot4_pass_heading)

        # Stop shortly after the tip crosses the puck center.  The target is
        # tied to the puck, rather than being an arbitrary long distance from
        # the current tip position.
        follow_through = min(0.10, 0.5 * self.pass_stroke_distance)
        target_x = self.puck_x + follow_through * shot_ux
        target_y = self.puck_y + follow_through * shot_uy
        self.robot4_tip_target = (target_x, target_y)

        self.robot4_pass_required_progress = max(
            0.03,
            (target_x - tip_x) * shot_ux
            + (target_y - tip_y) * shot_uy,
        )
        self.state_start_ns = self.get_clock().now().nanoseconds

        self.get_logger().info(
            f'Robot 4 straight pass initialized: '
            f'tip_start=({tip_x:.3f},{tip_y:.3f}); '
            f'target=({target_x:.3f},{target_y:.3f}); '
            f'body_heading={math.degrees(self.robot4_pass_heading):.1f} deg; '
            f'progress={self.robot4_pass_required_progress:.3f} m.'
        )

    def solve_robot4_tip_clf_cbf_qp(
        self,
        tip_x: float,
        tip_y: float,
        target_x: float,
        target_y: float,
        nominal_tip_ux: float,
        nominal_tip_uy: float,
    ) -> Tuple[float, float, bool, bool]:
        """CLF-CBF-QP in [v, omega] for Robot 4's physical stick tip."""
        obstacles, emergency_stop = self.collect_active_obstacles(
            self.robot4,
            tip_x,
            tip_y,
        )
        if emergency_stop:
            return 0.0, 0.0, False, True

        jacobian = self.robot4_stick_tip_jacobian()
        try:
            nominal_vw = np.linalg.solve(
                jacobian,
                np.array([nominal_tip_ux, nominal_tip_uy], dtype=float),
            )
        except np.linalg.LinAlgError:
            nominal_vw = np.linalg.lstsq(
                jacobian,
                np.array([nominal_tip_ux, nominal_tip_uy], dtype=float),
                rcond=None,
            )[0]

        nominal_v = clamp(
            float(nominal_vw[0]),
            -self.contact_max_v,
            self.contact_max_v,
        )
        nominal_w = clamp(
            float(nominal_vw[1]),
            -self.contact_max_w,
            self.contact_max_w,
        )

        # z = [v, omega, delta]
        p_matrix = np.diag([
            2.0,
            2.0,
            2.0 * self.clf_slack_weight,
        ])
        q_vector = np.array([
            -2.0 * nominal_v,
            -2.0 * nominal_w,
            0.0,
        ])

        g_rows = []
        h_values = []

        ex = tip_x - target_x
        ey = tip_y - target_y
        lyapunov_value = 0.5 * (ex * ex + ey * ey)
        clf_v = ex * jacobian[0, 0] + ey * jacobian[1, 0]
        clf_w = ex * jacobian[0, 1] + ey * jacobian[1, 1]
        g_rows.append([clf_v, clf_w, -1.0])
        h_values.append(-self.clf_rate * lyapunov_value)

        # delta >= 0
        g_rows.append([0.0, 0.0, -1.0])
        h_values.append(0.0)

        for obstacle_x, obstacle_y, safe_distance in obstacles:
            dx = tip_x - obstacle_x
            dy = tip_y - obstacle_y
            barrier_value = (
                dx * dx + dy * dy - safe_distance * safe_distance
            )
            cbf_v = -2.0 * (
                dx * jacobian[0, 0] + dy * jacobian[1, 0]
            )
            cbf_w = -2.0 * (
                dx * jacobian[0, 1] + dy * jacobian[1, 1]
            )
            g_rows.append([cbf_v, cbf_w, 0.0])
            h_values.append(self.cbf_rate * barrier_value)

        # Direct actuator limits for the stick-tip shooting phase.
        g_rows.extend([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        h_values.extend([
            self.contact_max_v,
            self.contact_max_v,
            self.contact_max_w,
            self.contact_max_w,
        ])

        solution = None
        for solver_name in (
            self.qp_solver,
            'quadprog',
            'cvxopt',
            'osqp',
        ):
            try:
                solution = solve_qp(
                    P=p_matrix,
                    q=q_vector,
                    G=np.asarray(g_rows, dtype=float),
                    h=np.asarray(h_values, dtype=float),
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

        return (
            clamp(float(solution[0]), -self.contact_max_v, self.contact_max_v),
            clamp(float(solution[1]), -self.contact_max_w, self.contact_max_w),
            True,
            False,
        )

    def execute_robot4_tip_pass(self) -> Tuple[bool, float]:
        """Drive straight through the puck while holding the frozen heading."""
        if (
            self.robot4_tip_start is None
            or self.robot4_tip_target is None
            or self.robot4_pass_heading is None
            or self.robot4_pass_puck_start is None
        ):
            raise RuntimeError('Robot 4 straight pass was not initialized.')

        tip_x, tip_y = self.robot4_stick_tip_position()
        start_x, start_y = self.robot4_tip_start
        target_x, target_y = self.robot4_tip_target
        shot_ux = math.cos(self.robot4_pass_heading)
        shot_uy = math.sin(self.robot4_pass_heading)

        progress = (
            (tip_x - start_x) * shot_ux
            + (tip_y - start_y) * shot_uy
        )
        remaining = max(0.0, self.robot4_pass_required_progress - progress)

        # Safety constraints remain active for every obstacle except the target
        # puck, whose CBF is intentionally disabled in PASS_WITH_R4.
        px, py = self.controlled_point(self.robot4)
        _, emergency = self.collect_active_obstacles(
            self.robot4,
            px,
            py,
        )
        if emergency:
            self.publish_stop(self.robot4)
            return False, remaining

        heading_error = wrap_angle(
            self.robot4_pass_heading - self.robot4.theta
        )

        heading_gate = math.radians(8.0)
        linear_speed = self.contact_max_v
        if abs(heading_error) > heading_gate:
            linear_speed = 0.0
        elif remaining < 0.06:
            linear_speed *= max(0.35, remaining / 0.06)

        cmd = Twist()
        cmd.linear.x = clamp(linear_speed, 0.0, self.contact_max_v)
        cmd.angular.z = clamp(
            self.k_heading * heading_error,
            -self.contact_max_w,
            self.contact_max_w,
        )
        self.robot4_cmd_pub.publish(cmd)

        elapsed = (
            self.get_clock().now().nanoseconds - self.state_start_ns
        ) / 1e9

        puck_progress = (
            (self.puck_x - self.robot4_pass_puck_start[0]) * shot_ux
            + (self.puck_y - self.robot4_pass_puck_start[1]) * shot_uy
        )
        puck_was_shot = puck_progress >= self.puck_motion_epsilon

        finished = (
            progress >= self.robot4_pass_required_progress
            or puck_was_shot
            or elapsed >= min(self.contact_timeout, 2.0)
        )
        if finished:
            self.publish_stop(self.robot4)

        return finished, remaining

    def robot4_perpendicular_stage_point(
        self,
    ) -> Tuple[float, float]:
        """Return the controlled-point target for a straight shot to Robot 5.

        Robot 4's body faces the puck-to-Robot-5 direction.  The target is
        shifted sideways to compensate for the angled stick mount, placing the
        physical stick tip just behind the puck before the forward stroke.
        """
        receiver_x, receiver_y = self.live_receiver_position()
        shot_heading = self.heading_between(
            self.puck_x, self.puck_y, receiver_x, receiver_y
        )
        shot_ux = math.cos(shot_heading)
        shot_uy = math.sin(shot_heading)

        # Desired initial stick-tip position, just behind the puck.
        tip_gap = self.puck_radius + self.robot4_tip_precontact_gap
        desired_tip_x = self.puck_x - tip_gap * shot_ux
        desired_tip_y = self.puck_y - tip_gap * shot_uy

        # navigate_robot_to() controls the near-identity point.  Subtract the
        # mounted-stick vector so that, after body alignment, the physical tip
        # occupies the desired pre-contact position.
        stick_heading = shot_heading + self.robot4_stick_angle_offset
        stage_x = (
            desired_tip_x
            - self.stick_tip_offset_from_point * math.cos(stick_heading)
        )
        stage_y = (
            desired_tip_y
            - self.stick_tip_offset_from_point * math.sin(stick_heading)
        )

        self.get_logger().info(
            f'Robot 4 compensated pass stage: '
            f'controlled_point=({stage_x:.3f},{stage_y:.3f}); '
            f'desired_tip=({desired_tip_x:.3f},{desired_tip_y:.3f}); '
            f'body_heading={math.degrees(shot_heading):.1f} deg.'
        )
        return stage_x, stage_y

    def robot4_shot_heading(self) -> float:
        """Return the live desired puck-travel heading toward Robot 5."""
        receiver_x, receiver_y = self.live_receiver_position()
        return self.heading_between(
            self.puck_x,
            self.puck_y,
            receiver_x,
            receiver_y,
        )

    def update_robot4_swing_direction_from_geometry(self) -> None:
        """
        Select clockwise or counterclockwise shooting automatically.

        The chosen sign makes the tangential velocity of the stick tip
        point from Robot 4 toward Robot 5. 
        """
        receiver_x, receiver_y = self.live_receiver_position()

        radius_x = self.puck_x - self.robot4.x
        radius_y = self.puck_y - self.robot4.y

        # Desired puck travel is from the puck toward Robot 5.
        target_x = receiver_x - self.puck_x
        target_y = receiver_y - self.puck_y

        # For positive (counterclockwise) angular velocity, the tangential
        # stick-tip velocity is (-radius_y, radius_x). Choose the sign whose
        # tangent points toward Robot 5.
        ccw_alignment = (
            -radius_y * target_x
            + radius_x * target_y
        )

        self.robot4_swing_direction = (
            1 if ccw_alignment >= 0.0 else -1
        )

        direction_name = (
            'counterclockwise'
            if self.robot4_swing_direction > 0
            else 'clockwise'
        )
        self.get_logger().info(
            f'Robot 4 selected {direction_name} shooting rotation '
            f'for the upper perpendicular staging point.'
        )

    def begin_robot4_ccw_preload(self) -> None:
        """Start a guaranteed counterclockwise preload from the live heading."""
        self.robot4_preload_start_heading = self.robot4.theta
        self.state_start_ns = self.get_clock().now().nanoseconds
        self.get_logger().warning(
            'Robot 4 PRELOAD START: commanding CCW rotation '
            f'for {math.degrees(self.robot4_backswing_angle):.1f} deg '
            f'at {self.robot4_backswing_max_w:.2f} rad/s.'
        )

    def execute_robot4_ccw_preload(self) -> Tuple[bool, float]:
        """Command positive angular velocity until the full CCW angle is measured."""
        if self.robot4_preload_start_heading is None:
            raise RuntimeError('Robot 4 CCW preload was not initialized.')

        ccw_progress = wrap_angle(
            self.robot4.theta - self.robot4_preload_start_heading
        )
        ccw_progress = max(0.0, ccw_progress)

        elapsed = (
            self.get_clock().now().nanoseconds - self.state_start_ns
        ) / 1e9
        preload_timeout = max(
            2.0,
            2.0 * self.robot4_backswing_angle
            / max(self.robot4_backswing_max_w, 1e-6),
        )

        finished = (
            ccw_progress >= self.robot4_backswing_angle
            or elapsed >= preload_timeout
        )

        if finished:
            self.publish_stop(self.robot4)
            self.get_logger().warning(
                'Robot 4 PRELOAD COMPLETE: '
                f'CCW progress={math.degrees(ccw_progress):.1f} deg. '
                'Next command will be full-speed clockwise.'
            )
            return True, ccw_progress

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = abs(self.robot4_backswing_max_w) 
        self.robot4_cmd_pub.publish(cmd)
        return False, ccw_progress

    def robot4_pass_impact_heading(self) -> float:
        """Return the body heading where the physical stick reaches the puck.

        The stick is mounted at robot4_stick_angle_offset from the chassis.
        Therefore the chassis impact heading equals the live Robot-4-to-puck
        bearing minus the mounting angle.
        """
        world_stick_heading = self.heading_between(
            self.robot4.x,
            self.robot4.y,
            self.puck_x,
            self.puck_y,
        )
        return wrap_angle(
            world_stick_heading - self.robot4_stick_angle_offset
        )

    def robot4_backswing_heading(self) -> float:
        """Compatibility helper: desired heading is explicitly CCW."""
        return wrap_angle(
            self.robot4.theta + self.robot4_backswing_angle
        )

    def begin_robot4_rotational_pass(self) -> None:
        """Initialize a guaranteed full-speed clockwise strike."""
        self.robot4_swing_direction = -1  # Negative angular.z is clockwise.
        self.robot4_swing_start_heading = self.robot4.theta
        self.robot4_impact_heading = self.robot4_pass_impact_heading()
        self.robot4_swing_total_angle = (
            self.robot4_backswing_angle
            + self.robot4_follow_through_angle
        )

        receiver_x, receiver_y = self.live_receiver_position()
        shot_x = receiver_x - self.puck_x
        shot_y = receiver_y - self.puck_y
        radius_x = self.puck_x - self.robot4.x
        radius_y = self.puck_y - self.robot4.y

        tangent_x = self.robot4_swing_direction * (-radius_y)
        tangent_y = self.robot4_swing_direction * radius_x
        tangent_norm = max(math.hypot(tangent_x, tangent_y), 1e-9)
        shot_norm = max(math.hypot(shot_x, shot_y), 1e-9)
        alignment_cos = clamp(
            (tangent_x * shot_x + tangent_y * shot_y)
            / (tangent_norm * shot_norm),
            -1.0,
            1.0,
        )
        tangent_error = math.degrees(math.acos(alignment_cos))

        world_stick_heading = wrap_angle(
            self.robot4_impact_heading
            + self.robot4_stick_angle_offset
        )
        self.get_logger().info(
            f'Robot 4 CW FULL-SPEED STRIKE initialized: '
            f'impact_body_heading='
            f'{math.degrees(self.robot4_impact_heading):.1f} deg; '
            f'impact_stick_heading='
            f'{math.degrees(world_stick_heading):.1f} deg; '
            f'backswing={math.degrees(self.robot4_backswing_angle):.1f} deg; '
            f'follow_through='
            f'{math.degrees(self.robot4_follow_through_angle):.1f} deg; '
            f'tangent_to_R5_error={tangent_error:.1f} deg; '
            f'commanded_omega_from_first_cycle=-{self.robot4_swing_max_w:.2f} rad/s; '
            f'max_omega={self.robot4_swing_max_w:.2f} rad/s; '
            f'max_omega_after='
            f'{math.degrees(self.robot4_preimpact_accel_fraction * self.robot4_backswing_angle):.1f} deg.'
        )

        self.state_start_ns = (
            self.get_clock().now().nanoseconds
        )

    def execute_robot4_rotational_pass(
        self,
    ) -> Tuple[bool, float, float]:
        """Rotate Robot 4 in place with an angular-speed ramp."""
        if self.robot4_swing_start_heading is None:
            raise RuntimeError(
                'Robot 4 swing start heading was not initialized.'
            )

        elapsed = (
            self.get_clock().now().nanoseconds
            - self.state_start_ns
        ) / 1e9

        angular_progress = (
            self.robot4_swing_direction
            * wrap_angle(
                self.robot4.theta
                - self.robot4_swing_start_heading
            )
        )
        angular_progress = max(0.0, angular_progress)

        # Full strike speed is commanded from the first clockwise control cycle.
        angular_speed = self.robot4_swing_max_w

        puck_motion = 0.0
        if self.previous_puck_position is not None:
            puck_motion = math.hypot(
                self.puck_x - self.previous_puck_position[0],
                self.puck_y - self.previous_puck_position[1],
            )
        puck_was_hit = (
            angular_progress >= 0.5 * self.robot4_backswing_angle
            and puck_motion >= self.puck_motion_epsilon
        )

        finished = (
            angular_progress >= self.robot4_swing_total_angle
            or puck_was_hit
            or elapsed >= self.robot4_swing_timeout
        )

        if finished:
            self.publish_stop(self.robot4)
            return True, angular_progress, 0.0

        cmd = Twist()
        cmd.linear.x = self.robot4_swing_forward_speed
        cmd.angular.z = (
            self.robot4_swing_direction
            * angular_speed
        )
        self.robot4_cmd_pub.publish(cmd)

        return (
            False,
            angular_progress,
            cmd.angular.z,
        )

    # ================================================================
    # State machine
    # ================================================================
    def transition_to(self, new_state: str) -> None:
        self.stop_both()
        self.state = new_state
        self.state_start_ns = self.get_clock().now().nanoseconds
        self.contact_start_point = None
        self.contact_target_point = None
        self.get_logger().info(f'T4 state -> {new_state}')

    def control_loop(self) -> None:
        if self.state == self.COMPLETE:
            self.stop_both()
            self.handoff_complete = True
            return

        if not (
            self.robot4.pose_received
            and self.robot5.pose_received
            and self.puck_pose_received
        ):
            self.stop_both()
            self.log_periodically(
                'Waiting for Robot 4, Robot 5, and '
                'green-puck poses...',
                'last_waiting_log_ns',
                2.0,
            )
            return

        if self.state == self.WAIT_FOR_POSES:
            self.receive_x, self.receive_y = (
                self.calculate_receiver_position()
            )
            self.receive_position_frozen = True
            self.transition_to(self.MOVE_R5_TO_RECEIVE)
            return

        # Periodic debug output.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_debug_log_ns >= int(0.5e9):
            self.get_logger().info(
                f'State={self.state}; '
                f'R4=({self.robot4.x:.2f},{self.robot4.y:.2f}); '
                f'R5=({self.robot5.x:.2f},{self.robot5.y:.2f}); '
                f'puck=({self.puck_x:.2f},{self.puck_y:.2f}); '
                f'desired_receive=({self.receive_x:.2f},'
                f'{self.receive_y:.2f}); '
                f'live_receiver=({self.robot5.x:.2f},'
                f'{self.robot5.y:.2f})'
            )
            self.last_debug_log_ns = now_ns

        if self.state == self.MOVE_R5_TO_RECEIVE:
            self.publish_stop(self.robot4)

            if not self.receiver_departure_aligned:
                departure_heading = self.heading_between(
                    self.robot5.x,
                    self.robot5.y,
                    self.receive_x,
                    self.receive_y,
                )
                heading_error = wrap_angle(
                    departure_heading - self.robot5.theta
                )

                if (
                    abs(heading_error)
                    > self.receiver_departure_heading_tolerance
                ):
                    cmd = Twist()
                    cmd.angular.z = clamp(
                        self.k_heading * heading_error,
                        -self.receiver_departure_max_w,
                        self.receiver_departure_max_w,
                    )
                    self.robot5_cmd_pub.publish(cmd)
                    return

                self.publish_stop(self.robot5)
                self.receiver_departure_aligned = True
                self.get_logger().info(
                    'Robot 5 departure heading aligned. '
                    'Starting translation to the receiving pose.'
                )

            success, error = self.navigate_robot_to(
                self.robot5,
                self.receive_x,
                self.receive_y,
            )
            if success and error <= self.position_tolerance:
                self.transition_to(self.ALIGN_R5_TO_RECEIVE)
            return

        if self.state == self.ALIGN_R5_TO_RECEIVE:
            self.publish_stop(self.robot4)
            incoming_heading = self.heading_between(
                self.receive_x,
                self.receive_y,
                self.puck_x,
                self.puck_y,
            )
            desired_heading = wrap_angle(
                incoming_heading + self.receiver_heading_offset
            )
            aligned, _ = self.align_robot(
                self.robot5,
                desired_heading,
            )
            if aligned:
                self.robot4_stage_target = (
                    self.robot4_perpendicular_stage_point()
                )
                self.transition_to(self.MOVE_R4_TO_PASS_STAGE)
            return

        if self.state == self.MOVE_R4_TO_PASS_STAGE:
            self.publish_stop(self.robot5)

            # Use the frozen behind-the-puck staging point. Recomputing it
            # every cycle could move the target because of mocap noise.
            if self.robot4_stage_target is None:
                self.robot4_stage_target = (
                    self.robot4_perpendicular_stage_point()
                )
            stage_x, stage_y = self.robot4_stage_target

            if not self.robot4_departure_aligned:
                departure_heading = self.heading_between(
                    self.robot4.x,
                    self.robot4.y,
                    stage_x,
                    stage_y,
                )

                heading_error = wrap_angle(
                    departure_heading - self.robot4.theta
                )

                if (
                    abs(heading_error)
                    > self.robot4_departure_heading_tolerance
                ):
                    cmd = Twist()
                    cmd.angular.z = clamp(
                        self.k_heading * heading_error,
                        -self.robot4_departure_max_w,
                        self.robot4_departure_max_w,
                    )
                    self.robot4_cmd_pub.publish(cmd)
                    return

                self.publish_stop(self.robot4)
                self.robot4_departure_aligned = True
                self.get_logger().info(
                    'Robot 4 departure heading aligned. '
                    'Starting translation to the pass stage.'
                )

            success, error = self.navigate_robot_to(
                self.robot4,
                stage_x,
                stage_y,
            )
            if success and error <= self.robot4_pass_stage_tolerance:
                self.get_logger().info(
                    f'Robot 4 reached the useful pass stage: '
                    f'error={error:.3f} m; '
                    f'tolerance={self.robot4_pass_stage_tolerance:.3f} m. '
                    'Starting the heading-corrected forward approach now.'
                )

                # The former PRECONTACT_R4 phase could spend its entire timeout
                # correcting heading while measured forward progress remained
                # zero.  The manual-forward controller already separates heading
                # alignment from measured translation, so enter it directly.
                self.transition_to(self.MANUAL_FORWARD_R4)
                self.begin_robot4_manual_forward()
            return

        if self.state == self.ALIGN_R4_TO_PASS:
            self.publish_stop(self.robot5)

            # The compensated stage places the angled stick tip behind the
            # puck. Align the robot body with the puck-to-Robot-5 shot line.
            aligned, _ = self.align_robot(
                self.robot4,
                self.robot4_pass_impact_heading(),
                max_w=self.robot4_backswing_max_w,
            )

            if aligned:
                tip_x, tip_y = self.robot4_stick_tip_position()
                tip_center_distance = math.hypot(
                    tip_x - self.puck_x,
                    tip_y - self.puck_y,
                )
                surface_gap = tip_center_distance - self.puck_radius
                self.get_logger().info(
                    'Robot 4 is staged behind the green puck and its body '
                    'is aligned for a straight shot toward Robot 5. '
                    f'Modeled stick-tip surface gap={surface_gap:.3f} m.'
                )
                self.transition_to(self.MANUAL_FORWARD_R4)
                self.begin_robot4_manual_forward()
            return

        if self.state == self.PRECONTACT_R4:
            # Legacy recovery path only. Normal execution no longer enters this
            # state because its timeout included heading-correction time.
            self.publish_stop(self.robot5)

            (
                finished,
                longitudinal_gap,
                lateral_error,
                remaining_advance,
            ) = self.execute_robot4_precontact_adjustment()

            if (
                self.get_clock().now().nanoseconds
                - self.last_debug_log_ns
                >= int(0.25e9)
            ):
                self.get_logger().info(
                    f'Robot 4 measured straight pre-contact: '
                    f'longitudinal_gap={longitudinal_gap:.3f} m; '
                    f'lateral_error={lateral_error:.3f} m; '
                    f'remaining_advance={remaining_advance:.3f} m'
                )
                self.last_debug_log_ns = (
                    self.get_clock().now().nanoseconds
                )

            if finished:
                self.transition_to(self.MANUAL_FORWARD_R4)
                self.begin_robot4_manual_forward()
            return

        if self.state == self.MANUAL_FORWARD_R4:
            self.publish_stop(self.robot5)

            finished, succeeded, progress, remaining = (
                self.execute_robot4_manual_forward()
            )

            if (
                self.get_clock().now().nanoseconds
                - self.last_debug_log_ns
                >= int(0.25e9)
            ):
                self.get_logger().info(
                    f'Robot 4 manual forward nudge: '
                    f'progress={progress:.3f} m; '
                    f'remaining={remaining:.3f} m'
                )
                self.last_debug_log_ns = (
                    self.get_clock().now().nanoseconds
                )

            if finished and succeeded:
                # Explicit required sequence:
                # 1) positive angular.z = counterclockwise preload
                # 2) negative angular.z = full-speed clockwise strike
                self.robot4_swing_direction = -1
                self.transition_to(self.BACKSWING_R4)
                self.begin_robot4_ccw_preload()
            elif finished:
                self.stop_both()
            return

        if self.state == self.BACKSWING_R4:
            self.publish_stop(self.robot5)

            preload_finished, preload_progress = (
                self.execute_robot4_ccw_preload()
            )

            if (
                self.get_clock().now().nanoseconds
                - self.last_debug_log_ns
                >= int(0.20e9)
            ):
                self.get_logger().info(
                    'Robot 4 CCW preload: '
                    f'progress={math.degrees(preload_progress):.1f} deg; '
                    f'target={math.degrees(self.robot4_backswing_angle):.1f} deg; '
                    f'commanded_omega=+{self.robot4_backswing_max_w:.2f} rad/s'
                )
                self.last_debug_log_ns = (
                    self.get_clock().now().nanoseconds
                )

            if preload_finished:
                self.transition_to(self.PASS_WITH_R4)
                self.begin_robot4_rotational_pass()
            return

        if self.state == self.PASS_WITH_R4:
            self.publish_stop(self.robot5)

            (
                finished,
                angular_progress,
                commanded_omega,
            ) = self.execute_robot4_rotational_pass()

            if (
                self.get_clock().now().nanoseconds
                - self.last_debug_log_ns
                >= int(0.25e9)
            ):
                self.get_logger().info(
                    f'Robot 4 rotational strike: '
                    f'progress={math.degrees(angular_progress):.1f} deg; '
                    f'target='
                    f'{math.degrees(self.robot4_swing_total_angle):.1f} deg; '
                    f'omega={commanded_omega:.3f} rad/s; '
                    f'impact_at='
                    f'{math.degrees(self.robot4_backswing_angle):.1f} deg'
                )
                self.last_debug_log_ns = (
                    self.get_clock().now().nanoseconds
                )

            if finished:
                self.puck_receive_counter = 0
                self.transition_to(self.WAIT_FOR_PUCK_AT_R5)
            return

        if self.state == self.WAIT_FOR_PUCK_AT_R5:
            self.stop_both()

            receiver_x, receiver_y = self.live_receiver_position()

            receive_error = math.hypot(
                self.puck_x - receiver_x,
                self.puck_y - receiver_y,
            )

            puck_motion = float('inf')
            if self.previous_puck_position is not None:
                puck_motion = math.hypot(
                    self.puck_x - self.previous_puck_position[0],
                    self.puck_y - self.previous_puck_position[1],
                )

            if (
                receive_error <= self.puck_receive_tolerance
                and puck_motion <= self.puck_motion_epsilon
            ):
                self.puck_receive_counter += 1
            else:
                self.puck_receive_counter = 0

            if (
                self.puck_receive_counter
                >= self.puck_receive_required_cycles
            ):
                self.transition_to(self.MOVE_R5_TO_GOAL_STAGE)
            return

        if self.state == self.MOVE_R5_TO_GOAL_STAGE:
            self.publish_stop(self.robot4)
            stage_x, stage_y, _ = self.stage_point_behind_puck(
                self.goal_x,
                self.goal_y,
            )
            success, error = self.navigate_robot_to(
                self.robot5,
                stage_x,
                stage_y,
            )
            if success and error <= self.position_tolerance:
                self.transition_to(self.ALIGN_R5_TO_GOAL)
            return

        if self.state == self.ALIGN_R5_TO_GOAL:
            self.publish_stop(self.robot4)
            goal_heading = self.heading_between(
                self.puck_x,
                self.puck_y,
                self.goal_x,
                self.goal_y,
            )
            aligned, _ = self.align_robot(
                self.robot5,
                goal_heading,
            )
            if aligned:
                self.transition_to(self.SHOOT_WITH_R5)
                self.begin_contact_stroke(
                    self.robot5,
                    goal_heading,
                    self.shot_stroke_distance,
                )
            return

        if self.state == self.SHOOT_WITH_R5:
            self.publish_stop(self.robot4)
            finished, _ = self.execute_contact_stroke(self.robot5)
            if finished:
                self.transition_to(self.COMPLETE)
            return

        self.stop_both()
        self.get_logger().error(f'Unknown state: {self.state}')
        self.transition_to(self.COMPLETE)

    # ================================================================
    # Logging
    # ================================================================
    def log_periodically(
        self,
        message: str,
        attribute_name: str,
        period_seconds: float,
    ) -> None:
        now_ns = self.get_clock().now().nanoseconds
        previous_ns = getattr(self, attribute_name)
        if now_ns - previous_ns >= int(period_seconds * 1e9):
            self.get_logger().warning(message)
            setattr(self, attribute_name, now_ns)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = T4PassAndShoot()

    try:
        while rclpy.ok() and not node.handoff_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_both()
        except Exception:
            pass

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()