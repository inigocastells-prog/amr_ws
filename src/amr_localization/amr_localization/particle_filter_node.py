import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

import message_filters
from amr_msgs.msg import PoseStamped, RangeScan
from nav_msgs.msg import Odometry

import os
import time
import traceback
from transforms3d.euler import euler2quat

from amr_localization.particle_filter import ParticleFilter


class ParticleFilterNode(LifecycleNode):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("particle_filter")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_plot", False)
        self.declare_parameter("particles", 3000)
        self.declare_parameter("steps_btw_sense_updates", 10)
        self.declare_parameter("world", "project")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            dt = self.get_parameter("dt").get_parameter_value().double_value
            self._enable_plot = self.get_parameter("enable_plot").get_parameter_value().bool_value
            particles = self.get_parameter("particles").get_parameter_value().integer_value
            steps_btw_sense_updates = (
                self.get_parameter("steps_btw_sense_updates").get_parameter_value().integer_value
            )
            world = self.get_parameter("world").get_parameter_value().string_value

            # Subscribers
            self._subscribers: list[message_filters.Subscriber] = []
            self._subscribers.append(message_filters.Subscriber(self, Odometry, "odom"))
            self._subscribers.append(message_filters.Subscriber(self, RangeScan, "us_scan"))

            ts = message_filters.ApproximateTimeSynchronizer(
                self._subscribers, queue_size=10, slop=2
            )
            ts.registerCallback(self._compute_pose_callback)

            # Publishers
            # TODO: 2.1. Create the /pose publisher (PoseStamped message).
            self.pose_pub = self.create_publisher(
                msg_type=PoseStamped, topic="/pose", qos_profile=10
            )
            # Constants
            SENSOR_RANGE = 1.0  # Ultrasonic sensor range [m]

            # Sensor location and orientation (x, y, theta) [m, m, rad] in the robot coordinate frame
            SENSORS = [
                (0.1067, 0.1382, 1.5708),
                (0.1557, 0.1250, 0.8727),
                (0.1909, 0.0831, 0.5236),
                (0.2095, 0.0273, 0.1745),
                (0.2095, -0.0273, -0.1745),
                (0.1909, -0.0785, -0.5236),
                (0.1558, -0.1203, -0.8727),
                (0.1067, -0.1382, -1.5708),
                (-0.1100, -0.1382, -1.5708),
                (-0.1593, -0.1203, -2.2689),
                (-0.1943, -0.0785, -2.6180),
                (-0.2129, -0.0273, -2.9671),
                (-0.2129, 0.0273, 2.9671),
                (-0.1943, 0.0785, 2.6180),
                (-0.1593, 0.1203, 2.2689),
                (-0.1100, 0.1382, 1.5708),
            ]

            # Attribute and object initializations
            self._localized = False
            self._log_level = self.get_logger().get_effective_level()
            self._steps = 0
            self._steps_btw_sense_updates = steps_btw_sense_updates
            map_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
            )
            self._particle_filter = ParticleFilter(
                dt, map_path, SENSORS, SENSOR_RANGE, particle_count=particles
            )

            if self._enable_plot:
                self._particle_filter.show("Initialization", save_figure=True)

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

    def _compute_pose_callback(self, odom_msg: Odometry, us_msg: RangeScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            us_msg: Message containing US sensor readings.

        """
        # Parse measurements
        z_us: list[float] = us_msg.ranges
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z

        # Execute particle filter
        self._execute_motion_step(z_v, z_w)
        x_h, y_h, theta_h = self._execute_measurement_step(z_us)
        self._steps += 1

        # Publish
        self._publish_pose_estimate(x_h, y_h, theta_h)

    def _execute_measurement_step(self, z_us: list[float]) -> tuple[float, float, float]:
        """Executes and monitors the measurement step (sense) of the particle filter.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].

        Returns:
            Pose estimate (x_h, y_h, theta_h) [m, m, rad]; inf if cannot be computed.
        """
        pose = (float("inf"), float("inf"), float("inf"))

        if self._localized or not self._steps % self._steps_btw_sense_updates:
            start_time = time.perf_counter()
            self._particle_filter.resample(z_us)
            sense_time = time.perf_counter() - start_time

            self.get_logger().info(f"Sense step time: {sense_time:6.3f} s")

            if self._enable_plot:
                self._particle_filter.show("Sense", save_figure=True)

            start_time = time.perf_counter()
            self._localized, pose = self._particle_filter.compute_pose()
            clustering_time = time.perf_counter() - start_time

            self.get_logger().info(f"Clustering time: {clustering_time:6.3f} s")

        return pose

    def _execute_motion_step(self, z_v: float, z_w: float):
        """Executes and monitors the motion step (move) of the particle filter.

        Args:
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].
        """
        start_time = time.perf_counter()
        self._particle_filter.move(z_v, z_w)
        move_time = time.perf_counter() - start_time

        self.get_logger().info(f"Move step time: {move_time:7.3f} s")

        if self._enable_plot:
            self._particle_filter.show("Move", save_figure=True)

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        """Publishes the robot's pose estimate in a custom amr_msgs.msg.PoseStamped message.

        Args:
            x_h: x coordinate estimate [m].
            y_h: y coordinate estimate [m].
            theta_h: Heading estimate [rad].

        """
        # TODO: 2.2. Complete the function body with your code (i.e., replace the pass statement).
        msg = PoseStamped()

        # check if localized
        msg.localized = self._localized
        if not self._localized:
            msg.pose.position.x = 0.0
            msg.pose.position.y = 0.0
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
            self.pose_pub.publish(msg)
            return
        
        # if localized 
        # store pose
        msg.pose.position.x = float(x_h)
        msg.pose.position.y = float(y_h)
        msg.pose.position.z = 0.0

        # calculate quaternions
        qx, qy, qz, qw = euler2quat(0.0, 0.0, float(theta_h))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        # publish
        self.pose_pub.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    particle_filter_node = ParticleFilterNode()

    try:
        rclpy.spin(particle_filter_node)
    except KeyboardInterrupt:
        pass

    particle_filter_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
