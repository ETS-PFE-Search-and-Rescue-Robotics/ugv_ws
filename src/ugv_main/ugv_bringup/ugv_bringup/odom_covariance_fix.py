import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

# Relays an odometry topic with realistic covariance values injected.
# rf2o and the wheel base node both publish near-zero covariance, which causes
# the EKF to treat measurements as infinitely certain and diverge.
# This node is launched twice: once for rf2o, once for wheel odom.


def _diag_to_cov(diag):
    cov = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 7] = v
    return cov


class OdomCovarianceFix(Node):
    def __init__(self):
        super().__init__('odom_covariance_fix')
        self.declare_parameter('input_topic', '/odom_rf2o')
        self.declare_parameter('output_topic', '/odom_rf2o_fixed')
        self.declare_parameter('pose_cov_diag', [0.05, 0.05, 1e6, 1e6, 1e6, 0.05])
        self.declare_parameter('twist_cov_diag', [0.05, 1e6,  1e6, 1e6, 1e6, 0.05])

        in_topic   = self.get_parameter('input_topic').value
        out_topic  = self.get_parameter('output_topic').value
        pose_diag  = list(self.get_parameter('pose_cov_diag').value)
        twist_diag = list(self.get_parameter('twist_cov_diag').value)

        self.pose_cov  = _diag_to_cov(pose_diag)
        self.twist_cov = _diag_to_cov(twist_diag)

        self.pub = self.create_publisher(Odometry, out_topic, 10)
        self.sub = self.create_subscription(Odometry, in_topic, self._cb, 10)

    def _cb(self, msg: Odometry):
        msg.pose.covariance  = self.pose_cov
        msg.twist.covariance = self.twist_cov
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = OdomCovarianceFix()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
