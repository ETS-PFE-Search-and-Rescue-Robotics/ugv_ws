"""
Utility functions for pose detection and analysis.

This module contains helper functions and utilities used throughout
the PoseSense library for pose processing and analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import math


class PoseUtils:
    """Utility class for pose-related calculations and operations."""
    
    @staticmethod
    def calculate_distance_2d(point1: Tuple[float, float], 
                             point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two 2D points.
        
        Args:
            point1: First point as (x, y) tuple
            point2: Second point as (x, y) tuple
            
        Returns:
            Distance between the points
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_distance_3d(point1: Tuple[float, float, float], 
                             point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D points.
        
        Args:
            point1: First point as (x, y, z) tuple
            point2: Second point as (x, y, z) tuple
            
        Returns:
            Distance between the points
        """
        return math.sqrt((point1[0] - point2[0])**2 + 
                        (point1[1] - point2[1])**2 + 
                        (point1[2] - point2[2])**2)
    
    @staticmethod
    def calculate_angle_between_vectors(vector1: Tuple[float, float], 
                                      vector2: Tuple[float, float]) -> float:
        """
        Calculate angle between two 2D vectors.
        
        Args:
            vector1: First vector as (x, y) tuple
            vector2: Second vector as (x, y) tuple
            
        Returns:
            Angle in degrees
        """
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        mag1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        mag2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
        
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
    
    @staticmethod
    def normalize_landmarks(landmarks: List[Dict[str, float]], 
                          image_width: int, 
                          image_height: int) -> List[Dict[str, float]]:
        """
        Normalize landmark coordinates to image dimensions.
        
        Args:
            landmarks: List of landmark dictionaries
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            List of normalized landmark dictionaries
        """
        normalized = []
        for landmark in landmarks:
            normalized.append({
                'x': landmark['x'] * image_width,
                'y': landmark['y'] * image_height,
                'z': landmark['z'],
                'visibility': landmark['visibility']
            })
        return normalized
    
    @staticmethod
    def get_bounding_box(landmarks: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
        """
        Calculate bounding box for pose landmarks.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        visible_landmarks = [lm for lm in landmarks if lm['visibility'] > 0.5]
        
        if not visible_landmarks:
            return (0.0, 0.0, 0.0, 0.0)
        
        x_coords = [lm['x'] for lm in visible_landmarks]
        y_coords = [lm['y'] for lm in visible_landmarks]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    @staticmethod
    def smooth_landmarks(current_landmarks: List[Dict[str, float]], 
                        previous_landmarks: List[Dict[str, float]], 
                        smoothing_factor: float = 0.7) -> List[Dict[str, float]]:
        """
        Apply temporal smoothing to landmarks.
        
        Args:
            current_landmarks: Current frame landmarks
            previous_landmarks: Previous frame landmarks
            smoothing_factor: Smoothing factor (0.0 to 1.0)
            
        Returns:
            Smoothed landmarks
        """
        if not previous_landmarks or len(current_landmarks) != len(previous_landmarks):
            return current_landmarks
        
        smoothed = []
        for curr, prev in zip(current_landmarks, previous_landmarks):
            smoothed_landmark = {
                'x': smoothing_factor * prev['x'] + (1 - smoothing_factor) * curr['x'],
                'y': smoothing_factor * prev['y'] + (1 - smoothing_factor) * curr['y'],
                'z': smoothing_factor * prev['z'] + (1 - smoothing_factor) * curr['z'],
                'visibility': curr['visibility']
            }
            smoothed.append(smoothed_landmark)
        
        return smoothed
    
    @staticmethod
    def filter_landmarks_by_visibility(landmarks: List[Dict[str, float]], 
                                     min_visibility: float = 0.5) -> List[Dict[str, float]]:
        """
        Filter landmarks by visibility threshold.
        
        Args:
            landmarks: List of landmark dictionaries
            min_visibility: Minimum visibility threshold
            
        Returns:
            Filtered landmarks
        """
        return [lm for lm in landmarks if lm['visibility'] >= min_visibility]
    
    @staticmethod
    def draw_skeleton(image: np.ndarray, 
                     landmarks: List[Dict[str, float]], 
                     connections: List[Tuple[int, int]], 
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw skeleton connections on image.
        
        Args:
            image: Input image
            landmarks: List of landmark dictionaries
            connections: List of landmark index pairs to connect
            color: Line color in BGR format
            thickness: Line thickness
            
        Returns:
            Image with skeleton drawn
        """
        h, w = image.shape[:2]
        result_image = image.copy()
        
        for connection in connections:
            start_idx, end_idx = connection
            
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx]['visibility'] > 0.5 and
                landmarks[end_idx]['visibility'] > 0.5):
                
                start_point = (int(landmarks[start_idx]['x'] * w),
                             int(landmarks[start_idx]['y'] * h))
                end_point = (int(landmarks[end_idx]['x'] * w),
                           int(landmarks[end_idx]['y'] * h))
                
                cv2.line(result_image, start_point, end_point, color, thickness)
        
        return result_image
    
    @staticmethod
    def get_pose_center(landmarks: List[Dict[str, float]]) -> Tuple[float, float]:
        """
        Calculate the center point of the pose.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Center point as (x, y) tuple
        """
        visible_landmarks = [lm for lm in landmarks if lm['visibility'] > 0.5]
        
        if not visible_landmarks:
            return (0.5, 0.5)
        
        center_x = sum(lm['x'] for lm in visible_landmarks) / len(visible_landmarks)
        center_y = sum(lm['y'] for lm in visible_landmarks) / len(visible_landmarks)
        
        return (center_x, center_y)
    
    @staticmethod
    def scale_landmarks_to_image(landmarks: List[Dict[str, float]], 
                               image_shape: Tuple[int, int]) -> List[Dict[str, float]]:
        """
        Scale normalized landmarks to image pixel coordinates.
        
        Args:
            landmarks: List of normalized landmark dictionaries
            image_shape: Image shape as (height, width)
            
        Returns:
            Scaled landmarks
        """
        height, width = image_shape
        scaled = []
        
        for landmark in landmarks:
            scaled.append({
                'x': landmark['x'] * width,
                'y': landmark['y'] * height,
                'z': landmark['z'],
                'visibility': landmark['visibility']
            })
        
        return scaled
    
    @staticmethod
    def create_pose_heatmap(landmarks: List[Dict[str, float]], 
                          image_shape: Tuple[int, int], 
                          radius: int = 10) -> np.ndarray:
        """
        Create a heatmap visualization of pose landmarks.
        
        Args:
            landmarks: List of landmark dictionaries
            image_shape: Target image shape as (height, width)
            radius: Radius for landmark points
            
        Returns:
            Heatmap image
        """
        height, width = image_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for landmark in landmarks:
            if landmark['visibility'] > 0.5:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                # Create a small circle around the landmark
                y_start = max(0, y - radius)
                y_end = min(height, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(width, x + radius + 1)
                
                for py in range(y_start, y_end):
                    for px in range(x_start, x_end):
                        dist = math.sqrt((px - x)**2 + (py - y)**2)
                        if dist <= radius:
                            intensity = (1.0 - dist / radius) * landmark['visibility']
                            heatmap[py, px] = max(heatmap[py, px], intensity)
        
        # Convert to 8-bit and apply colormap
        heatmap_8bit = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        return heatmap_colored