from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ep_hockey_controller',
            executable='t1_navigate',
            name='t1_navigate',
            output='screen',
            parameters=[{
                'robot_id': 5,

                'stick_topic':
                    '/vrpn_mocap/hockey_sticks_1/pose',

                'stick_side': 'right',
            }],
        ),
    ])