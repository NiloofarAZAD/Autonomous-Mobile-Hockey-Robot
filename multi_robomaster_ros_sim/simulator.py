import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import ColorRGBA, Bool
from math import cos, sin, pi
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class MultiRoboMasterSim(Node):
    def __init__(self):
        super().__init__('multi_robomaster_sim')
        # constants
        # robots
        self.ROBOT_IDS = [4, 5]
        self.N = len(self.ROBOT_IDS)
        # time
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 20 # milliseconds
        self.TIMEOUT_GET_POSES = 10 # milliseconds
        self.TIMEOUT_CHASSIS_SPEED = 500 # milliseconds
        self.DT = (self.TIMEOUT_SET_MOBILE_BASE_SPEED + self.TIMEOUT_GET_POSES) / 1000.
        # robot control
        self.MAX_LINEAR_SPEED = 1.0 # meters / second
        self.MAX_ANGULAR_SPEED = 360 * np.pi / 180 # radians / second
        # dimensions
        self.ENV = [-2., -2., 4., 4.] # (x, y) can vary from (ENV[0], ENV[1]) to (ENV[0]+ENV[2], ENV[1]+ENV[3])
        self.ROBOT_SIZE = [0.24, 0.32] # [w, l]
        self.GRIPPER_SIZE = 0.1

        # Stick attachment state.
        # T1 publishes True on /simulator/attach_stick when pickup is complete.
        self.STICK_LENGTH = 0.22

        self.left_stick_attached_to = None
        self.left_stick_local_x = 0.0
        self.left_stick_local_y = 0.0
        self.left_stick_theta_offset = 0.0

        self.right_stick_attached_to = None
        self.right_stick_local_x = 0.0
        self.right_stick_local_y = 0.0
        self.right_stick_theta_offset = 0.0

        # State: [x, y, theta]
        self.states = {}
        self.leds = {}
        self.velocities = {rid: np.array([0.0, 0.0, 0.0]) for rid in self.ROBOT_IDS}
        self.last_cmd_time = {rid: self.get_clock().now() for rid in self.ROBOT_IDS}

        # Initialize robots at fixed positions.
        initial_states = {
            4: np.array([-0.95, 0.45, 2.36]),
            5: np.array([1.0, 0.5, 4.63]),
        }

        for i, rid in enumerate(self.ROBOT_IDS):
            self.states[rid] = initial_states[rid].copy()
            self.leds[rid] = np.array([0., 0., 0.])
        # Pubs and Subs
        self.pubs = {}
        self.subs_vel = {}
        self.subs_led = {}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        for rid in self.ROBOT_IDS:
            # Publisher: Mimics VRPN motion capture system
            self.pubs[rid] = self.create_publisher(
                PoseStamped, f'/vrpn_mocap/dji_robot_{rid}/pose', qos)

            # Subscriber: Listen to the controller's cmd_vel
            self.subs_vel[rid] = self.create_subscription(
                Twist, f'/robot{rid}/cmd_vel',
                lambda msg, rid=rid: self.vel_callback(msg, rid), qos)
            # Subscriber: Listen to the controller's leds
            self.subs_led[rid] = self.create_subscription(
                ColorRGBA, f'/robot{rid}/leds/color',
                lambda msg, rid=rid: self.led_callback(msg, rid), qos)

        # Publishers for the displayed hockey objects.
        self.red_puck_pub = self.create_publisher(
            PoseStamped,
            '/vrpn_mocap/hockey_puck_red/pose',
            qos
        )
        self.green_puck_pub = self.create_publisher(
            PoseStamped,
            '/vrpn_mocap/hockey_puck_green/pose',
            qos
        )
        self.green_puck_motion_sub = self.create_subscription(
            PoseStamped,
            '/sim/hockey_puck_green/pose',
            self.green_puck_motion_callback,
            qos
        )
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/vrpn_mocap/hockey_goal_1/pose',
            qos
        )
        self.left_stick_pub = self.create_publisher(
            PoseStamped,
            '/vrpn_mocap/hockey_sticks_1/pose',
            qos,
        )

        self.right_stick_pub = self.create_publisher(
            PoseStamped,
            '/vrpn_mocap/hockey_sticks_2/pose',
            qos,
        )

        # Receive the attachment command sent by the T1 controller.
        self.left_stick_attach_sub = self.create_subscription(
            Bool,
            '/simulator/attach_left_stick',
            self.left_stick_attach_callback,
            10,
        )

        self.right_stick_attach_sub = self.create_subscription(
            Bool,
            '/simulator/attach_right_stick',
            self.right_stick_attach_callback,
            10,
        )

        self.timer = self.create_timer(self.DT, self.update_and_publish)
        self.get_logger().info(f"Simulator started for robots: {self.ROBOT_IDS}")
        # Plots
        self.figure = []
        self.axes = []
        self.patches_robots = {rid: [] for rid in self.ROBOT_IDS}
        self.patches_grippers = {rid: [] for rid in self.ROBOT_IDS}
        self.text_ids = {rid: [] for rid in self.ROBOT_IDS}
        self.__init_plot()
        self.__update_plot()

    def __init_plot(self):
        self.figure, self.axes = plt.subplots(figsize=(10, 6))
        self.figure.subplots_adjust(right=0.75)
        p_env = patches.Rectangle(np.array([self.ENV[0], self.ENV[1]]), self.ENV[2], self.ENV[3], edgecolor=(0, 0, 0, 1), fill=False, linewidth=4)
        self.axes.add_patch(p_env)

        # Static hockey objects.
        # Stick holder at the bottom center.
        holder_width = 0.60
        holder_height = 0.16
        holder_x = -holder_width / 2.0
        holder_y = self.ENV[1] + 0.08
        p_holder = patches.Rectangle(
            (holder_x, holder_y),
            holder_width,
            holder_height,
            facecolor='#4f81bd',
            edgecolor='#17365d',
            linewidth=2
        )
        self.axes.add_patch(p_holder)

        # Two minimal small sticks represented by black lines.
        
        stick_bottom = holder_y + holder_height
        # Left stick
        self.left_stick_x = -0.08
        self.left_stick_y = stick_bottom
        self.left_stick_theta = pi / 2.0

        self.left_stick_line, = self.axes.plot(
            [self.left_stick_x, self.left_stick_x],
            [
                self.left_stick_y,
                self.left_stick_y + self.STICK_LENGTH,
            ],
            color='#17365d',
            linewidth=2,
        )

        # Right stick
        self.right_stick_x = 0.08
        self.right_stick_y = stick_bottom
        self.right_stick_theta = pi / 2.0

        self.right_stick_line, = self.axes.plot(
            [self.right_stick_x, self.right_stick_x],
            [
                self.right_stick_y,
                self.right_stick_y + self.STICK_LENGTH,
            ],
            color="#a41111",
            linewidth=2,
        )

        # Random puck positions.
        puck_margin = 0.20

        self.p1_x = -1.0
        self.p1_y = -0.51

        self.p2_x = 0.0
        self.p2_y = 0.0

        # Blue puck P1.
        p_puck_1 = patches.Circle(
            (self.p1_x, self.p1_y),
            radius=0.10,
            facecolor='red',
            edgecolor='none',
            linewidth=0
        )
        self.axes.add_patch(p_puck_1)
        self.axes.text(
            self.p1_x,
            self.p1_y - 0.17,
            'P1',
            color='black',
            fontsize=9,
            fontweight='normal',
            ha='center',
            va='top'
        )

        # Green puck P2.
        p_puck_2 = patches.Circle(
            (self.p2_x, self.p2_y),
            radius=0.10,
            facecolor='green',
            edgecolor='none',
            linewidth=0
        )
        self.green_puck_patch = p_puck_2
        self.axes.add_patch(p_puck_2)
        self.axes.text(
            self.p2_x,
            self.p2_y - 0.17,
            'P2',
            color='black',
            fontsize=9,
            fontweight='normal',
            ha='center',
            va='top'
        )

        # Orange rectangular goal at the right-center.
        goal_width = 0.16
        goal_height = 0.60
        goal_x = self.ENV[0] + self.ENV[2] - goal_width - 0.12
        goal_y = -goal_height / 2.0

        # Publish the center of the displayed rectangular goal.
        self.goal_x = goal_x + goal_width / 2.0
        self.goal_y = goal_y + goal_height / 2.0
        self.goal_theta = 0.0

        p_goal = patches.Rectangle(
            (goal_x, goal_y),
            goal_width,
            goal_height,
            facecolor='orange',
            edgecolor='darkorange',
            linewidth=2
        )
        self.axes.add_patch(p_goal)
        # self.axes.text(
        #     goal_x + 0.25,
        #     goal_y + goal_height + 0.08,
        #     'Goal',
        #     color='black',
        #     fontsize=9,
        #     fontweight='normal',
        #     ha='right',
        #     va='bottom'
        # )

        for i, rid in enumerate(self.ROBOT_IDS):
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])], [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            p_robot = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                                     [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T),
                                                     facecolor='k')
            p_gripper = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T),
                                                       facecolor='k')
            text_id = plt.text(self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0, self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0, s=str(self.ROBOT_IDS[i]), color="red")
            self.patches_robots[rid] = p_robot
            self.patches_grippers[rid] = p_gripper
            self.text_ids[rid] = text_id
            self.axes.add_patch(p_robot)
            self.axes.add_patch(p_gripper)
        self.axes.set_xlim(self.ENV[0] - max(self.ROBOT_SIZE), self.ENV[0] + self.ENV[2] + max(self.ROBOT_SIZE))
        self.axes.set_ylim(self.ENV[1] - max(self.ROBOT_SIZE), self.ENV[1] + self.ENV[3] + max(self.ROBOT_SIZE))
        self.axes.grid()
        # self.axes.set_axis_off()
        self.axes.axis('equal')

        legend_elements = [
            patches.Patch(
                facecolor='black',
                edgecolor='black',
                label='Robot'
            ),

            Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor='red',
                markeredgecolor='none',
                markersize=14,
                label='Red Puck (P1)'
            ),

            Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor='green',
                markeredgecolor='none',
                markersize=14,
                label='Green Puck (P2)'
            ),

            patches.Patch(
                facecolor='orange',
                edgecolor='darkorange',
                label='Goal'
            ),

            patches.Patch(
                facecolor='#4f81bd',
                edgecolor='#17365d',
                label='Rigid Body'
            ),

            Line2D(
                [0], [0],
                color='#17365d',
                linewidth=2,
                label='Left Stick'
            ),

            Line2D(
                [0], [0],
                color='#a41111',
                linewidth=2,
                label='Right Stick'
            ),
        ]

        self.axes.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.03, 0.5),
            frameon=True,
            borderaxespad=0.0,

            fontsize=11,

            borderpad=1.2,
            labelspacing=1.2,
            handlelength=2.2,
            handleheight=1.3,
            handletextpad=0.9
        )

        plt.ion()
        plt.show()

    def __update_plot(self):
        for rid in self.ROBOT_IDS:
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])], [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            xy_robot = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                      [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T)
            xy_gripper = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T)

            self.patches_robots[rid].xy = xy_robot
            self.patches_grippers[rid].xy = xy_gripper
            self.patches_robots[rid].set_facecolor(self.leds[rid])

            self.text_ids[rid].set_position((self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0, self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0))

        # Update the left stick line.
        left_end_x = (
            self.left_stick_x
            + self.STICK_LENGTH * cos(self.left_stick_theta)
        )
        left_end_y = (
            self.left_stick_y
            + self.STICK_LENGTH * sin(self.left_stick_theta)
        )

        self.left_stick_line.set_data(
            [self.left_stick_x, left_end_x],
            [self.left_stick_y, left_end_y],
        )

        # Update the right stick line.
        right_end_x = (
            self.right_stick_x
            + self.STICK_LENGTH * cos(self.right_stick_theta)
        )
        right_end_y = (
            self.right_stick_y
            + self.STICK_LENGTH * sin(self.right_stick_theta)
        )

        self.right_stick_line.set_data(
            [self.right_stick_x, right_end_x],
            [self.right_stick_y, right_end_y],
        )

        self.green_puck_patch.center = (
            self.p2_x,
            self.p2_y,
        )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    @staticmethod
    def transform_velocity_local_to_global(robots_speeds, theta):
        # robots_speeds : list of 3
        # theta : scalar
        robots_speeds_global = [0] * 3
        x_dot = robots_speeds[0]
        y_dot = robots_speeds[1]
        th_dot = robots_speeds[2]
        c_th = cos(theta)
        s_th = sin(theta)
        robots_speeds_global[0] = c_th * x_dot - s_th * y_dot
        robots_speeds_global[1] = s_th * x_dot + c_th * y_dot
        robots_speeds_global[2] = robots_speeds[2]
        return robots_speeds_global
    def vel_callback(self, msg, rid):
        # Store commanded velocities
        robot_speeds = MultiRoboMasterSim.transform_velocity_local_to_global([msg.linear.x, msg.linear.y, msg.angular.z], self.states[rid][2])
        self.velocities[rid] = np.array(robot_speeds)
        self.last_cmd_time[rid] = self.get_clock().now() # Update heartbeat
    def led_callback(self, msg, rid):
        # Store commanded velocities
        self.leds[rid] = np.array([msg.r, msg.g, msg.b])

    def green_puck_motion_callback(self, msg):
        self.p2_x = msg.pose.position.x
        self.p2_y = msg.pose.position.y

    def left_stick_attach_callback(self, msg):
        if not msg.data or self.left_stick_attached_to is not None:
            return

        rid = 4
        robot_x, robot_y, robot_theta = self.states[rid]

        dx = self.left_stick_x - robot_x
        dy = self.left_stick_y - robot_y
        c = cos(robot_theta)
        s = sin(robot_theta)

        self.left_stick_local_x = c * dx + s * dy
        self.left_stick_local_y = -s * dx + c * dy
        self.left_stick_theta_offset = (
            self.left_stick_theta - robot_theta
        )
        self.left_stick_attached_to = rid

        self.get_logger().info('Left stick attached to robot 4.')


    def right_stick_attach_callback(self, msg):
        if not msg.data or self.right_stick_attached_to is not None:
            return

        rid = 5
        robot_x, robot_y, robot_theta = self.states[rid]

        dx = self.right_stick_x - robot_x
        dy = self.right_stick_y - robot_y
        c = cos(robot_theta)
        s = sin(robot_theta)

        self.right_stick_local_x = c * dx + s * dy
        self.right_stick_local_y = -s * dx + c * dy
        self.right_stick_theta_offset = (
            self.right_stick_theta - robot_theta
        )
        self.right_stick_attached_to = rid

        self.get_logger().info('Right stick attached to robot 5.')

    def update_attached_sticks(self):
        if self.left_stick_attached_to is not None:
            rid = self.left_stick_attached_to
            robot_x, robot_y, robot_theta = self.states[rid]
            c = cos(robot_theta)
            s = sin(robot_theta)

            self.left_stick_x = (
                robot_x
                + c * self.left_stick_local_x
                - s * self.left_stick_local_y
            )
            self.left_stick_y = (
                robot_y
                + s * self.left_stick_local_x
                + c * self.left_stick_local_y
            )
            self.left_stick_theta = (
                robot_theta + self.left_stick_theta_offset
            )

        if self.right_stick_attached_to is not None:
            rid = self.right_stick_attached_to
            robot_x, robot_y, robot_theta = self.states[rid]
            c = cos(robot_theta)
            s = sin(robot_theta)

            self.right_stick_x = (
                robot_x
                + c * self.right_stick_local_x
                - s * self.right_stick_local_y
            )
            self.right_stick_y = (
                robot_y
                + s * self.right_stick_local_x
                + c * self.right_stick_local_y
            )
            self.right_stick_theta = (
                robot_theta + self.right_stick_theta_offset
            )

    def publish_object_pose(self, publisher, x, y, theta=0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        half_yaw = theta * 0.5
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = sin(half_yaw)
        msg.pose.orientation.w = cos(half_yaw)

        publisher.publish(msg)

    def update_and_publish(self):
        current_time = self.get_clock().now()
        for rid in self.ROBOT_IDS:
            elapsed_time_since_last_command_received = (current_time - self.last_cmd_time[rid]).nanoseconds / 1e9
            if elapsed_time_since_last_command_received > self.TIMEOUT_CHASSIS_SPEED / 1e3:
                v_cmd = np.array([0.0, 0.0, 0.0])
            else:
                v_cmd = self.velocities[rid]

            # Integrate velocity
            # Global X/Y update
            # The controller sends local velocities, which is what the robots are expected to receive.
            # These are then converted to global in the velocity callback in the simulator
            self.states[rid][0] += v_cmd[0] * self.DT
            self.states[rid][1] += v_cmd[1] * self.DT
            self.states[rid][2] += v_cmd[2] * self.DT
            # Create PoseStamped message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'

            msg.pose.position.x = self.states[rid][0]
            msg.pose.position.y = self.states[rid][1]
            msg.pose.position.z = 0.0

            # Euler to Quaternion (simplified for 2D Z-axis rotation)
            half_yaw = self.states[rid][2] * 0.5
            msg.pose.orientation.z = sin(half_yaw)
            msg.pose.orientation.w = cos(half_yaw)

            self.pubs[rid].publish(msg)

        # After attachment, move and rotate the stick with robot 4.
        self.update_attached_sticks()

        # Publish the exact poses of the displayed hockey objects.
        self.publish_object_pose(
            self.red_puck_pub,
            self.p1_x,
            self.p1_y
        )
        self.publish_object_pose(
            self.green_puck_pub,
            self.p2_x,
            self.p2_y
        )
        self.publish_object_pose(
            self.goal_pub,
            self.goal_x,
            self.goal_y,
            self.goal_theta
        )

        self.publish_object_pose(
            self.left_stick_pub,
            self.left_stick_x,
            self.left_stick_y,
            self.left_stick_theta,
        )

        self.publish_object_pose(
            self.right_stick_pub,
            self.right_stick_x,
            self.right_stick_y,
            self.right_stick_theta,
        )

        self.__update_plot()

def main(args=None):
    rclpy.init(args=args)
    node = MultiRoboMasterSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
