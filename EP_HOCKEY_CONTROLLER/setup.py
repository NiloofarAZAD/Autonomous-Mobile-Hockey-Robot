import os
from glob import glob
from setuptools import find_packages, setup


package_name = 'ep_hockey_controller'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        (
            'share/' + package_name,
            ['package.xml']
        ),

        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='root',
    maintainer_email='root@todo.todo',

    description='RoboMaster EP hockey controller',

    license='TODO',

    entry_points={
        'console_scripts': [
            't1_navigate = ep_hockey_controller.t1_navigate:main',
            't2_pickup = ep_hockey_controller.t2_pickup:main',
            't3_navigate = ep_hockey_controller.t3_navigate:main',
            't4_pass_and_shoot = ep_hockey_controller.t4_pass_and_shoot:main',

        ],
    },
)