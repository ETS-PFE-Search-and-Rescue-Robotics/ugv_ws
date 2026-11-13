import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
#from ugv_interface.action import Behavior
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from ugv_lying_detection.vendor.pose_sense.detector import PoseDetector
from ugv_lying_detection.vendor.pose_sense.classifier import ActivityClassifier
import cv2
import math

# How far to stop in front of the detected body (meters)
APPROACH_MARGIN = 0.4


class LyingPoseDetector(Node):
    def __init__(self):
        super().__init__('lying_pose_detector')
        # Create a subscription to the image_raw topic
        self.image_raw_subscription = self.create_subscription(Image,'/image_raw', self.image_callback,10)
        # Create a subscription to the scan topic
        self.lidar_subscription = self.create_subscription(LaserScan,'/scan', self.scan_callback,10)
        # Create an action client to send goals to the behavior action server
        #self._action_client = ActionClient(self, Behavior, 'behavior')
        # --- NEW: publisher for Nav2 goals + TF listener to know where we are ---
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_sent = False
        # Create a CvBridge object to convert between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

        # Create an ActionClient
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose') 

        # Create a PoseSense pose detector
        self.pose_detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Create a PoseSense pose classifier
        self.activity_classifier = ActivityClassifier()

        self.found_lying_human = False
        self.distance = None

    def image_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = self.pose_detector.detect_pose(image_rgb)

        if pose_result:
            angles = self.pose_detector.get_body_angles(pose_result)
            classification = self.activity_classifier.classify_activity(pose_result, angles)
            
            self.found_lying_human = (classification['activity'] == "lying")

        if self.found_lying_human:
            # Get the Lidar distance from the lying human
            self.get_logger().info("Found lying human")

            if self.distance is not None:
                self.get_logger().info(f"Human detected at {self.distance:.2f}m")
                # --- NEW: advance toward the body once ---
                if not self.goal_sent:
                    self.send_advance_goal(self.distance - APPROACH_MARGIN)
            else:
                self.get_logger().info("No LiDAR reading - skipping")

        # Convert the image back to a ROS Image message
        result_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

    def send_advance_goal(self, forward_dist):
        """Navigate to a point `forward_dist` meters ahead of the robot (map frame)."""
        if forward_dist <= 0.0:
            self.get_logger().info("Body already within approach margin, not moving.")
            self.goal_sent = True
            return
        #try:
        #    tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        #except Exception as e:
        #    self.get_logger().info(f"No map->base_link TF yet ({e}); goal not sent. Is localization running?")
        #    return

        #x = tf.transform.translation.x
        #y = tf.transform.translation.y
        #q = tf.transform.rotation
        #yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Target point: forward_dist ahead along current heading
        #tx = x + forward_dist * math.cos(yaw)
        #ty = y + forward_dist * math.sin(yaw)

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        #goal.pose.header.frame_id = 'map'
        goal.pose.header.frame_id = 'base_link'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        #goal.pose.pose.position.x = tx
        goal.pose.pose.position.x = float(forward_dist)
        #goal.pose.pose.position.y = ty
        #goal.pose.pose.position.y = float(forward_dist)
        #goal.pose.pose.orientation.z = math.sin(yaw / 2.0)   # keep current heading at arrival
        #goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.pose.pose.orientation.w = 1.0

        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("navigate_to_pose server not available (is slam_nav.launch.py running?)")
            return

        self._client.send_goal_async(goal, feedback_callback=self.fb).add_done_callback(self.on_response)
        self.goal_sent = True
        self.get_logger().info(f"Lying human at {self.distance:.2f}m -> navigating {forward_dist:.2f}m ahead to ({0:.2f}, {forward_dist:.2f})")

    def fb(self, msg):  
        self.get_logger().info(f"Remaining: {msg.feedback.distance_remaining:.2f} m")  
  
    def on_response(self, future):  
        handle = future.result()  
        if not handle.accepted:  
            self.get_logger().info('Goal rejected')  
            return  
        handle.get_result_async().add_done_callback(  
            lambda f: self.get_logger().info(f'Done: {f.result().status}'))


    def scan_callback(self, msg: LaserScan):
        if msg is None:
            return None
        
        forward_angle = msg.angle_max - msg.angle_min - (3/4) * math.pi
        index_per_rad = len(msg.ranges) / forward_angle
        
        # 0 radians = forward
        forward_idx = int(-msg.angle_min * index_per_rad)
        #forward_idx = 405
        
        # Sample a small window around forward
        start = max(0, forward_idx - 5)
        end = min(len(msg.ranges), forward_idx + 5)
        
        valid_dist = [dist for dist in msg.ranges[start:end] if math.isfinite(dist) and dist > 0.1]
        
        self.distance = min(valid_dist) if valid_dist else None

        #if self.distance is not None:
        #    print(f"\n### DISTANCE : {self.distance} ###\n")


def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)

    # Create a LyingPoseDetector node
    lying_pose_detector = LyingPoseDetector()
    # Spin the node
    rclpy.spin(lying_pose_detector)
    # Destroy the node
    lying_pose_detector.destroy_node()
    # Shutdown the ROS client library
    rclpy.shutdown()

if __name__ == '__main__':
    # Run the main function
    main()
