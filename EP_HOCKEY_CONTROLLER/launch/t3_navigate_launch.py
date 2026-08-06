from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ep_hockey_controller',
            executable='t3_navigate',
            name='t3_navigate',
            output='screen',
        ),
    ])