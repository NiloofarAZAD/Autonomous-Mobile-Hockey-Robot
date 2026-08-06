from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ep_hockey_controller',
            executable='t4_pass_and_shoot',
            name='t4_pass_and_shoot',
            output='screen',
        ),
    ])