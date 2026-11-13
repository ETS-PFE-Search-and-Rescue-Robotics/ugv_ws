"""
Pose Detection Module using MediaPipe

This module provides a wrapper around MediaPipe's pose detection functionality
to detect human body landmarks and poses from images or video streams.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class PoseDetector:
    """
    A wrapper class for MediaPipe pose detection.
    
    This class provides an easy-to-use interface for detecting human poses
    from images or video streams using Google's MediaPipe framework.
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the PoseDetector.
        
        Args:
            static_image_mode: Whether to treat input images as static images
            model_complexity: Complexity of the pose landmark model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to generate segmentation mask
            smooth_segmentation: Whether to smooth segmentation across frames
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_names = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
            'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT',
            'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
            'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
            'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
    
    def detect_pose(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect pose landmarks in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing pose landmarks and metadata, or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            return {
                'landmarks': landmarks,
                'landmark_names': self.landmark_names,
                'pose_landmarks': results.pose_landmarks,
                'pose_world_landmarks': results.pose_world_landmarks,
                'segmentation_mask': results.segmentation_mask
            }
        
        return None
    
    def get_landmark_coordinates(self, pose_result: Dict[str, Any], 
                               landmark_name: str) -> Optional[Tuple[float, float, float]]:
        """
        Get coordinates for a specific landmark.
        
        Args:
            pose_result: Result from detect_pose method
            landmark_name: Name of the landmark (e.g., 'LEFT_SHOULDER')
            
        Returns:
            Tuple of (x, y, z) coordinates, or None if landmark not found
        """
        if landmark_name not in self.landmark_names:
            return None
            
        landmark_idx = self.landmark_names.index(landmark_name)
        landmark = pose_result['landmarks'][landmark_idx]
        
        return (landmark['x'], landmark['y'], landmark['z'])
    
    def draw_pose(self, image: np.ndarray, pose_result: Dict[str, Any]) -> np.ndarray:
        """
        Draw pose landmarks on the image.
        
        Args:
            image: Input image as numpy array
            pose_result: Result from detect_pose method
            
        Returns:
            Image with pose landmarks drawn
        """
        annotated_image = image.copy()
        
        if pose_result and pose_result['pose_landmarks']:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_result['pose_landmarks'],
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return annotated_image
    
    def get_body_angles(self, pose_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate important body angles from pose landmarks.
        
        Args:
            pose_result: Result from detect_pose method
            
        Returns:
            Dictionary containing calculated angles
        """
        angles = {}
        
        if not pose_result:
            return angles
        
        # Helper function to calculate angle between three points
        def calculate_angle(point1: Tuple[float, float], 
                          point2: Tuple[float, float], 
                          point3: Tuple[float, float]) -> float:
            """Calculate angle at point2 formed by point1-point2-point3"""
            a = np.array(point1)
            b = np.array(point2)
            c = np.array(point3)
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360.0 - angle
            
            return angle
        
        try:
            # Get key landmarks
            left_shoulder = self.get_landmark_coordinates(pose_result, 'LEFT_SHOULDER')[:2]
            right_shoulder = self.get_landmark_coordinates(pose_result, 'RIGHT_SHOULDER')[:2]
            left_hip = self.get_landmark_coordinates(pose_result, 'LEFT_HIP')[:2]
            right_hip = self.get_landmark_coordinates(pose_result, 'RIGHT_HIP')[:2]
            left_knee = self.get_landmark_coordinates(pose_result, 'LEFT_KNEE')[:2]
            right_knee = self.get_landmark_coordinates(pose_result, 'RIGHT_KNEE')[:2]
            left_ankle = self.get_landmark_coordinates(pose_result, 'LEFT_ANKLE')[:2]
            right_ankle = self.get_landmark_coordinates(pose_result, 'RIGHT_ANKLE')[:2]
            
            # Calculate trunk angle (spine angle)
            if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                                 (left_shoulder[1] + right_shoulder[1]) / 2)
                hip_center = ((left_hip[0] + right_hip[0]) / 2,
                             (left_hip[1] + right_hip[1]) / 2)
                
                # Calculate angle from vertical
                trunk_angle = np.arctan2(abs(shoulder_center[0] - hip_center[0]),
                                       abs(shoulder_center[1] - hip_center[1])) * 180 / np.pi
                angles['trunk_angle'] = trunk_angle
            
            # Calculate leg angles
            if all([left_hip, left_knee, left_ankle]):
                angles['left_leg_angle'] = calculate_angle(left_hip, left_knee, left_ankle)
            
            if all([right_hip, right_knee, right_ankle]):
                angles['right_leg_angle'] = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angle (between thighs and trunk)
            if all([left_shoulder, left_hip, left_knee]):
                angles['left_hip_angle'] = calculate_angle(left_shoulder, left_hip, left_knee)
            
            if all([right_shoulder, right_hip, right_knee]):
                angles['right_hip_angle'] = calculate_angle(right_shoulder, right_hip, right_knee)
                
        except Exception as e:
            print(f"Error calculating angles: {e}")
        
        return angles
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()