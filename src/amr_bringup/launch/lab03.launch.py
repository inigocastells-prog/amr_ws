from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition

from amr_msgs.msg import PoseStamped
from rosidl_runtime_py import message_to_yaml

import math
from transforms3d.euler import euler2quat


def generate_launch_description():
    world = "project"
    start = (-4.0, -4.0, math.pi / 2)
    goal = (4.0, 4.0)

    msg = PoseStamped()
    msg.localized = True
    quat_w, quat_x, quat_y, quat_z = euler2quat(0.0, 0.0, start[2])
    msg.pose.position.x = start[0]
    msg.pose.position.y = start[1]
    msg.pose.orientation.w = quat_w
    msg.pose.orientation.x = quat_x
    msg.pose.orientation.y = quat_y
    msg.pose.orientation.z = quat_z

    a_star_node = LifecycleNode(
        package="amr_planning",
        executable="a_star",
        name="a_star",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "INFO"],
        parameters=[{"goal": goal, "world": world}],
    )

    lifecycle_manager_node = Node(
        package="amr_bringup",
        executable="lifecycle_manager",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"node_startup_order": ("a_star",)}],
    )

    publish_pose = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=a_star_node,
            goal_state="active",
            entities=[
                ExecuteProcess(
                    name="publish_pose",
                    cmd=[
                        "ros2",
                        "topic",
                        "pub",
                        "--once",
                        "/pose",
                        "amr_msgs/msg/PoseStamped",
                        message_to_yaml(msg),
                    ],
                    output="screen",
                )
            ],
        )
    )

    return LaunchDescription(
        [
            a_star_node,
            publish_pose,
            lifecycle_manager_node,  # Must be launched last
        ]
    )
