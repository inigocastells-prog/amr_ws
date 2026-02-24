import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

from amr_msgs.msg import PoseStamped as AmrPoseStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import os
import traceback

import numpy as np

from amr_planning.a_star import AStar


class AStarNode(LifecycleNode):
    def __init__(self):
        """A* node initializer."""
        super().__init__("a_star")

        # Parameters
        self.declare_parameter("goal", (0.0, 0.0))
        self.declare_parameter("world", "project")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            self._goal = tuple(
                self.get_parameter("goal").get_parameter_value().double_array_value.tolist()
            )
            world = self.get_parameter("world").get_parameter_value().string_value

            # Subscribers
            self._subscriber_pose = self.create_subscription(
                AmrPoseStamped, "pose", self._path_callback, 10
            )

            # Publishers
            # TODO: 3.5. Create the /path publisher (Path message).
            self._path_pub = self.create_publisher(Path, "/path", 10)

            # Constants
            SENSOR_RANGE = 1.0  # Ultrasonic sensor range [m]

            # Attribute and object initializations
            map_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
            )
            self._localized = False
            self._planning = AStar(map_path, SENSOR_RANGE, action_costs=(1.0, 1.0, 1.0, 1.0))

        except Exception:
            self.get_logger().error(f"{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles an activating transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)

    def _path_callback(self, pose_msg: AmrPoseStamped):
        """Subscriber callback. Executes A* and publishes the smoothed path to the goal.

        Args:
            pose_msg: Message containing the robot pose estimate.

        """
        if pose_msg.localized and not self._localized:
            # Execute A*
            start = (pose_msg.pose.position.x, pose_msg.pose.position.y)
            path, steps = self._planning.a_star(start, self._goal)
            smoothed_path = AStar.smooth_path(path, data_weight=0.3, smooth_weight=0.1)

            # --- Run naive search (heuristic = 0) ---
            original_compute_heuristic = self._planning._compute_heuristic

            # Override heuristic temporarily
            self._planning._compute_heuristic = lambda goal: \
                np.zeros_like(self._planning._map.grid_map)

            _, naive_steps = self._planning.a_star(start, self._goal)

            # Restore original heuristic
            self._planning._compute_heuristic = original_compute_heuristic


            self.get_logger().info(f"Path found in {steps} steps.")
            self.get_logger().info(f"Naive search (h=0) found path in {naive_steps} steps.")
            self._planning.show(path, smoothed_path, save_figure=True)
            self.get_logger().warn(f"Using action_costs = {self._planning._action_costs}")


            self._publish_path(smoothed_path)

        self._localized = pose_msg.localized

    def _publish_path(self, path: list[tuple[float, float]]) -> None:
        """Publishes the robot's path to the goal in a nav_msgs.msg.Path message.

        Args:
            path: Smoothed path (initial location first) in (x, y) format.

        """
        # TODO: 3.6. Complete the function body with your code (i.e., replace the pass statement).
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # si tu proyecto usa otro, cámbialo aquí

        poses: list[PoseStamped] = []

        for (x, y) in path:
            p = PoseStamped()
            p.header.stamp = msg.header.stamp
            p.header.frame_id = msg.header.frame_id

            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = 0.0

            # Orientación neutra (no se usa aún en Lab 3)
            p.pose.orientation.w = 1.0
            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = 0.0

            poses.append(p)

        msg.poses = poses
        self._path_pub.publish(msg)

        

def main(args=None):
    rclpy.init(args=args)

    a_star_node = AStarNode()

    try:
        rclpy.spin(a_star_node)
    except KeyboardInterrupt:
        pass

    a_star_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
