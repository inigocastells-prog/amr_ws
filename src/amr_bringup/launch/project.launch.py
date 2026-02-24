from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node

import math


def generate_launch_description():
    world = "project"
    start = (2.0, -3.0, 1.5 * math.pi)
    goal = (3.0, 2.0)
    particles = 10000

    wall_follower_node = LifecycleNode(
        package="amr_control",
        executable="wall_follower",
        name="wall_follower",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"enable_localization": True}],
    )

    particle_filter_node = LifecycleNode(
        package="amr_localization",
        executable="particle_filter",
        name="particle_filter",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"enable_plot": False, "particles": particles, "world": world}],
    )

    a_star_node = LifecycleNode(
        package="amr_planning",
        executable="a_star",
        name="a_star",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"goal": goal, "world": world}],
    )

    pure_pursuit_node = LifecycleNode(
        package="amr_control",
        executable="pure_pursuit",
        name="pure_pursuit",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
    )

    coppeliasim_node = LifecycleNode(
        package="amr_simulation",
        executable="coppeliasim",
        name="coppeliasim",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"enable_localization": True, "start": start, "goal": goal}],
    )

    lifecycle_manager_node = Node(
        package="amr_bringup",
        executable="lifecycle_manager",
        output="screen",
        arguments=["--ros-args", "--log-level", "INFO"],
        parameters=[
            {
                "node_startup_order": (
                    "wall_follower",
                    "particle_filter",
                    "a_star",
                    "pure_pursuit",
                    "coppeliasim",  # Must be started last
                )
            }
        ],
    )

    return LaunchDescription(
        [
            wall_follower_node,
            particle_filter_node,
            a_star_node,
            pure_pursuit_node,
            coppeliasim_node,
            lifecycle_manager_node,  # Must be launched last
        ]
    )
