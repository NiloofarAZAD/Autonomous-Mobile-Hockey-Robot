from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ep_hockey_controller',
            executable='t2_pickup',
            name='t2_pickup',
            output='screen',
            parameters=[{
                'robot_id': 5,
            }],
        ),
    ])