import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Whether to launch RViz2'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='bringup',
        description='Choose which rviz configuration to use'
    )
    use_ekf_arg = DeclareLaunchArgument(
        'use_ekf', default_value='false',
        description='Fuse rf2o through EKF (true) or use rf2o directly as /odom (false)'
    )

    use_ekf = LaunchConfiguration('use_ekf')

    robot_state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ugv_description'), 'launch', 'display.launch.py')
        ),
        launch_arguments={
            'use_rviz': LaunchConfiguration('use_rviz'),
            'rviz_config': LaunchConfiguration('rviz_config'),
        }.items()
    )

    bringup_node = Node(
        package='ugv_bringup',
        executable='ugv_bringup',
    )

    driver_node = Node(
        package='ugv_bringup',
        executable='ugv_driver',
    )

    laser_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ldlidar'), 'launch', 'ldlidar.launch.py')
        )
    )

    # use_ekf=true: EKF owns the odom→base_footprint TF, rf2o only publishes /odom_rf2o
    rf2o_with_ekf = Node(
        condition=IfCondition(use_ekf),
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic': '/odom_rf2o',
            'publish_tf': False,
            'base_frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            'init_pose_from_topic': '',
            'freq': 20.0,
        }],
    )

    # use_ekf=false: rf2o publishes TF and /odom directly, no EKF in the loop
    rf2o_standalone = Node(
        condition=UnlessCondition(use_ekf),
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic': '/odom',
            'publish_tf': True,
            'base_frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            'init_pose_from_topic': '',
            'freq': 20.0,
        }],
    )

    # Wheel encoder odometry — debug/reference only, not fused into anything
    base_node = Node(
        package='ugv_base_node',
        executable='base_node',
        parameters=[{'pub_odom_tf': False}],
        remappings=[('/odom', '/odom_wheel')]
    )

    # Covariance fix + EKF — only when use_ekf:=true
    odom_cov_fix_node = Node(
        condition=IfCondition(use_ekf),
        package='ugv_bringup',
        executable='odom_covariance_fix',
        name='rf2o_covariance_fix',
        output='screen',
        parameters=[{
            'input_topic':    '/odom_rf2o',
            'output_topic':   '/odom_rf2o_fixed',
            'pose_cov_diag':  [0.05, 0.05, 1e6, 1e6, 1e6, 0.05],
            'twist_cov_diag': [0.05, 1e6,  1e6, 1e6, 1e6, 0.05],
        }]
    )

    ekf_node = Node(
        condition=IfCondition(use_ekf),
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(get_package_share_directory('ugv_bringup'), 'param', 'ekf.yaml')],
        remappings=[('/odometry/filtered', '/odom')]
    )

    return LaunchDescription([
        use_rviz_arg,
        rviz_config_arg,
        use_ekf_arg,
        robot_state_launch,
        bringup_node,
        driver_node,
        laser_bringup_launch,
        # Delay rf2o and everything that depends on it by 3s so robot_state_publisher
        # has time to publish static TF (base_footprint→base_lidar_link) first.
        # Without this delay, rf2o gets its first scan before base_footprint exists
        # in the TF tree, fails silently, and never publishes odometry.
        TimerAction(period=3.0, actions=[
            rf2o_with_ekf,
            rf2o_standalone,
            odom_cov_fix_node,
            base_node,
            ekf_node,
        ]),
    ])
