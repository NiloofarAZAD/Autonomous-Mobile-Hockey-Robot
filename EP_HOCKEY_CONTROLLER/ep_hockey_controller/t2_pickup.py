# START
# move arm up to the stick height
# open gripper
# move arm slightly forward toward the stick
# close gripper around the stick
# lift arm slightly
# move robot backward approximately 0.40 m
# COMPLETE


import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node

from robomaster_msgs.action import GripperControl, MoveArm


class T2Pickup(Node):
    OPEN = 1
    CLOSE = 2

    def __init__(self):
        super().__init__('t2_pickup')

        # ------------------------------------------------------------
        # Declare parameters
        # ------------------------------------------------------------

        self.declare_parameter('robot_id', 5)
        self.declare_parameter('gripper_power', 0.5)

        # Absolute arm position before opening the gripper.
        # The arm first moves to approximately 5 cm in the arm z coordinate.
        self.declare_parameter('arm_ready_x', 0.0)
        self.declare_parameter('arm_ready_z', 0.05)

        # Relative forward motion after the gripper opens.
        self.declare_parameter('arm_forward_x', 0.12)
        self.declare_parameter('arm_forward_z', 0.0)

        # Relative lift after the gripper closes.
        # A positive z command raises the arm by 10 cm.
        self.declare_parameter('arm_lift_x', 0.0)
        self.declare_parameter('arm_lift_z', 0.10)

        # Give the gripper time to hold the stick before lifting.
        self.declare_parameter('gripper_settle_time', 1.0)

        self.declare_parameter('backward_speed', -0.10)
        self.declare_parameter('backward_distance', 0.40)

        # ------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------

        self.robot_id = int(
            self.get_parameter('robot_id').value
        )

        self.gripper_power = float(
            self.get_parameter('gripper_power').value
        )

        self.arm_ready_x = float(
            self.get_parameter('arm_ready_x').value
        )

        self.arm_ready_z = float(
            self.get_parameter('arm_ready_z').value
        )

        self.arm_forward_x = float(
            self.get_parameter('arm_forward_x').value
        )

        self.arm_forward_z = float(
            self.get_parameter('arm_forward_z').value
        )

        self.arm_lift_x = float(
            self.get_parameter('arm_lift_x').value
        )

        self.arm_lift_z = float(
            self.get_parameter('arm_lift_z').value
        )

        self.gripper_settle_time = float(
            self.get_parameter('gripper_settle_time').value
        )

        if self.gripper_settle_time < 0.0:
            raise ValueError(
                'gripper_settle_time cannot be negative.'
            )

        self.backward_speed = float(
            self.get_parameter('backward_speed').value
        )

        self.backward_distance = float(
            self.get_parameter('backward_distance').value
        )

        if self.backward_speed >= 0.0:
            raise ValueError(
                'backward_speed must be negative.'
            )

        if self.backward_distance <= 0.0:
            raise ValueError(
                'backward_distance must be positive.'
            )

        self.backward_duration = (
            self.backward_distance
            / abs(self.backward_speed)
        )

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------

        self.gripper_client = ActionClient(
            self,
            GripperControl,
            f'/robot{self.robot_id}/gripper'
        )

        self.arm_client = ActionClient(
            self,
            MoveArm,
            f'/robot{self.robot_id}/move_arm'
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            f'/robot{self.robot_id}/cmd_vel',
            10
        )

        # ------------------------------------------------------------
        # State machine
        # ------------------------------------------------------------

        self.state = 'START'
        self.sequence_started = False
        self.motion_timer = None
        self.backward_publish_timer = None
        self.gripper_settle_start_time = None

        self.timer = self.create_timer(
            0.1,
            self.state_machine
        )

        self.get_logger().info(
            f'T2 pickup controller started for robot '
            f'{self.robot_id}.'
        )

    # ================================================================
    # State machine
    # ================================================================

    def state_machine(self):
        if self.sequence_started:
            return

        if self.state == 'START':
            self.sequence_started = True

            self.get_logger().info(
                'Starting automatic pickup sequence.'
            )

            self.get_logger().info(
                'Moving arm up to the stick height first.'
            )

            self.move_arm_to_ready_position()

        elif self.state == 'MOVE_ARM_TO_STICK':
            self.sequence_started = True

            self.get_logger().info(
                'Moving arm toward stick pickup position.'
            )

            self.move_arm_to_stick()

        elif self.state == 'OPEN_GRIPPER':
            self.sequence_started = True

            self.get_logger().info(
                'Opening gripper at the stick pickup position.'
            )

            self.send_gripper_command(
                target_state=self.OPEN,
                next_state='MOVE_ARM_TO_STICK'
            )

        elif self.state == 'CLOSE_GRIPPER':
            self.sequence_started = True

            self.get_logger().info(
                'Closing gripper around the stick.'
            )

            self.send_gripper_command(
                target_state=self.CLOSE,
                next_state='WAIT_AFTER_GRIP'
            )

        elif self.state == 'WAIT_AFTER_GRIP':
            if self.gripper_settle_start_time is None:
                self.gripper_settle_start_time = (
                    self.get_clock().now()
                )

                self.get_logger().info(
                    'Stick gripped. Waiting briefly before lifting.'
                )

            elapsed = (
                self.get_clock().now()
                - self.gripper_settle_start_time
            ).nanoseconds / 1e9

            if elapsed >= self.gripper_settle_time:
                self.gripper_settle_start_time = None
                self.state = 'LIFT_ARM'
                self.sequence_started = False

            return

        elif self.state == 'LIFT_ARM':
            self.sequence_started = True

            self.get_logger().info(
                'Lifting the arm with the stick.'
            )

            self.lift_arm()

        elif self.state == 'MOVE_BACKWARD':
            self.sequence_started = True

            self.get_logger().info(
                'Moving backward to clear the pickup area.'
            )

            self.move_backward()

        elif self.state == 'COMPLETE':
            self.sequence_started = True
            self.stop_robot()

            self.get_logger().info(
                'T2 complete: stick pickup sequence finished.'
            )

    # ================================================================
    # Gripper action
    # ================================================================

    def send_gripper_command(
        self,
        target_state: int,
        next_state: str
    ):
        if not self.gripper_client.wait_for_server(
            timeout_sec=2.0
        ):
            self.get_logger().error(
                'Gripper action server is unavailable.'
            )

            self.sequence_started = False
            return

        goal = GripperControl.Goal()
        goal.target_state = int(target_state)
        goal.power = float(self.gripper_power)

        future = self.gripper_client.send_goal_async(goal)

        future.add_done_callback(
            lambda result_future:
                self.gripper_goal_response_callback(
                    result_future,
                    next_state
                )
        )

    def gripper_goal_response_callback(
        self,
        future,
        next_state: str
    ):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Failed to send gripper goal: {error}'
            )
            self.sequence_started = False
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                'Gripper goal was rejected.'
            )
            self.sequence_started = False
            return

        self.get_logger().info(
            'Gripper command accepted.'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            lambda completed_future:
                self.gripper_result_callback(
                    completed_future,
                    next_state
                )
        )

    def gripper_result_callback(
        self,
        future,
        next_state: str
    ):
        try:
            wrapped_result = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Gripper action failed: {error}'
            )
            self.sequence_started = False
            return

        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                'Gripper action did not succeed. '
                f'Status: {wrapped_result.status}'
            )
            self.sequence_started = False
            return

        duration = wrapped_result.result.duration

        self.get_logger().info(
            'Gripper command completed in '
            f'{duration.sec}.{duration.nanosec:09d} seconds.'
        )

        self.state = next_state
        self.sequence_started = False

    # ================================================================
    # Arm movement sequence
    # ================================================================

    def move_arm_to_ready_position(self):
        self.send_arm_command(
            x=self.arm_ready_x,
            z=self.arm_ready_z,
            relative=False,
            next_state='OPEN_GRIPPER'
        )

    def move_arm_to_stick(self):
        self.send_arm_command(
            x=self.arm_forward_x,
            z=self.arm_forward_z,
            relative=True,
            next_state='CLOSE_GRIPPER'
        )

    def lift_arm(self):
        self.send_arm_command(
            x=self.arm_lift_x,
            z=self.arm_lift_z,
            relative=True,
            next_state='MOVE_BACKWARD'
        )

    # ================================================================
    # Arm action
    # ================================================================

    def send_arm_command(
        self,
        x: float,
        z: float,
        relative: bool,
        next_state: str
    ):
        if not self.arm_client.wait_for_server(
            timeout_sec=2.0
        ):
            self.get_logger().error(
                'Arm action server is unavailable.'
            )

            self.sequence_started = False
            return

        goal = MoveArm.Goal()
        goal.x = float(x)
        goal.z = float(z)
        goal.relative = bool(relative)

        self.get_logger().info(
            f'Sending arm command: '
            f'x={goal.x:.3f} m, '
            f'z={goal.z:.3f} m, '
            f'relative={goal.relative}'
        )

        future = self.arm_client.send_goal_async(
            goal,
            feedback_callback=self.arm_feedback_callback
        )

        future.add_done_callback(
            lambda result_future:
                self.arm_goal_response_callback(
                    result_future,
                    next_state
                )
        )

    def arm_feedback_callback(self, feedback_msg):
        progress = feedback_msg.feedback.progress

        self.get_logger().info(
            f'Arm movement progress: '
            f'{progress * 100.0:.1f}%'
        )

    def arm_goal_response_callback(
        self,
        future,
        next_state: str
    ):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Failed to send arm goal: {error}'
            )
            self.sequence_started = False
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                'Arm goal was rejected.'
            )
            self.sequence_started = False
            return

        self.get_logger().info(
            'Arm command accepted.'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            lambda completed_future:
                self.arm_result_callback(
                    completed_future,
                    next_state
                )
        )

    def arm_result_callback(
        self,
        future,
        next_state: str
    ):
        try:
            wrapped_result = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Arm action failed: {error}'
            )
            self.sequence_started = False
            return

        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                'Arm action did not succeed. '
                f'Status: {wrapped_result.status}'
            )
            self.sequence_started = False
            return

        self.get_logger().info(
            'Arm movement completed.'
        )

        self.state = next_state
        self.sequence_started = False

    # ================================================================
    # Backward clearance movement
    # ================================================================

    def move_backward(self):
        self.publish_backward_command()

        self.backward_publish_timer = self.create_timer(
            0.10,
            self.publish_backward_command
        )

        self.create_motion_stop_timer(
            duration=self.backward_duration,
            next_state='COMPLETE'
        )

    def publish_backward_command(self):
        command = Twist()
        command.linear.x = self.backward_speed
        command.angular.z = 0.0
        self.cmd_vel_publisher.publish(command)

    def create_motion_stop_timer(
        self,
        duration: float,
        next_state: str
    ):
        self.motion_timer = self.create_timer(
            duration,
            lambda: self.finish_backward_motion(next_state)
        )

    def finish_backward_motion(
        self,
        next_state: str
    ):
        if self.motion_timer is not None:
            self.motion_timer.cancel()
            self.destroy_timer(self.motion_timer)
            self.motion_timer = None

        if self.backward_publish_timer is not None:
            self.backward_publish_timer.cancel()
            self.destroy_timer(self.backward_publish_timer)
            self.backward_publish_timer = None

        self.stop_robot()

        self.get_logger().info(
            'Backward clearance movement completed.'
        )

        self.state = next_state
        self.sequence_started = False

    def stop_robot(self):
        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0

        self.cmd_vel_publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)

    node = T2Pickup()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()