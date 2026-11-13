"""
Activity Classification Module

This module classifies human activities based on pose landmarks detected by MediaPipe.
It can identify activities such as sitting, lying down, standing, and other postures.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .utils import PoseUtils


class ActivityClassifier:
    """
    Classifies human activities based on pose landmarks.
    
    This class analyzes pose landmarks and body angles to determine
    what activity a person is performing (sitting, lying, standing, etc.).
    """
    
    def __init__(self):
        """Initialize the ActivityClassifier."""
        self.pose_utils = PoseUtils()
        
        # Activity thresholds and parameters
        self.thresholds = {
            'sitting': {
                'hip_angle_min': 70,
                'hip_angle_max': 120,
                'trunk_angle_max': 45,
                'leg_angle_min': 70,
                'leg_angle_max': 120
            },
            'lying': {
                'trunk_angle_min': 70,
                'hip_y_threshold': 0.7,  # Relative to image height
                'shoulder_hip_distance_max': 0.3
            },
            'standing': {
                'leg_angle_min': 160,
                'trunk_angle_max': 20,
                'hip_angle_min': 160
            },
            'squatting': {
                'hip_angle_max': 90,
                'leg_angle_max': 90,
                'trunk_angle_max': 45
            }
        }
        
        # Confidence thresholds
        self.min_visibility = 0.5
        self.min_confidence = 0.6
    
    def classify_activity(self, pose_result: Dict[str, Any], 
                         angles: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify the activity based on pose landmarks and angles.
        
        Args:
            pose_result: Result from PoseDetector.detect_pose()
            angles: Body angles from PoseDetector.get_body_angles()
            
        Returns:
            Dictionary containing activity classification results
        """
        if not pose_result or not pose_result.get('landmarks'):
            return {
                'activity': 'unknown',
                'confidence': 0.0,
                'details': 'No pose detected'
            }
        
        # Check landmark visibility
        if not self._check_landmark_visibility(pose_result):
            return {
                'activity': 'unknown',
                'confidence': 0.0,
                'details': 'Insufficient landmark visibility'
            }
        
        # Calculate activity scores
        activity_scores = {}
        
        # Check for sitting
        sitting_score = self._classify_sitting(pose_result, angles)
        activity_scores['sitting'] = sitting_score
        
        # Check for lying
        lying_score = self._classify_lying(pose_result, angles)
        activity_scores['lying'] = lying_score
        
        # Check for standing
        standing_score = self._classify_standing(pose_result, angles)
        activity_scores['standing'] = standing_score
        
        # Check for squatting
        squatting_score = self._classify_squatting(pose_result, angles)
        activity_scores['squatting'] = squatting_score
        
        # Determine best classification
        best_activity = max(activity_scores.keys(), key=lambda k: activity_scores[k])
        best_confidence = activity_scores[best_activity]
        
        # Apply minimum confidence threshold
        if best_confidence < self.min_confidence:
            best_activity = 'unknown'
            best_confidence = 0.0
        
        return {
            'activity': best_activity,
            'confidence': best_confidence,
            'all_scores': activity_scores,
            'details': f'Classified as {best_activity} with {best_confidence:.2f} confidence'
        }
    
    def _check_landmark_visibility(self, pose_result: Dict[str, Any]) -> bool:
        """Check if key landmarks are visible enough for classification."""
        key_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
                        'LEFT_KNEE', 'RIGHT_KNEE']
        
        landmark_names = pose_result['landmark_names']
        landmarks = pose_result['landmarks']
        
        visible_count = 0
        for landmark_name in key_landmarks:
            if landmark_name in landmark_names:
                idx = landmark_names.index(landmark_name)
                if landmarks[idx]['visibility'] >= self.min_visibility:
                    visible_count += 1
        
        return visible_count >= len(key_landmarks) * 0.7  # At least 70% visible
    
    def _classify_sitting(self, pose_result: Dict[str, Any], 
                         angles: Dict[str, float]) -> float:
        """Classify sitting posture."""
        score = 0.0
        criteria_met = 0
        total_criteria = 0
        
        # Check hip angles
        if 'left_hip_angle' in angles and 'right_hip_angle' in angles:
            hip_angle = (angles['left_hip_angle'] + angles['right_hip_angle']) / 2
            if (self.thresholds['sitting']['hip_angle_min'] <= hip_angle <= 
                self.thresholds['sitting']['hip_angle_max']):
                criteria_met += 1
            total_criteria += 1
        
        # Check leg angles (bent legs)
        if 'left_leg_angle' in angles and 'right_leg_angle' in angles:
            leg_angle = (angles['left_leg_angle'] + angles['right_leg_angle']) / 2
            if (self.thresholds['sitting']['leg_angle_min'] <= leg_angle <= 
                self.thresholds['sitting']['leg_angle_max']):
                criteria_met += 1
            total_criteria += 1
        
        # Check trunk angle (relatively upright)
        if 'trunk_angle' in angles:
            if angles['trunk_angle'] <= self.thresholds['sitting']['trunk_angle_max']:
                criteria_met += 1
            total_criteria += 1
        
        # Check relative positions (hips should be roughly at knee level or above)
        hip_knee_score = self._check_hip_knee_position(pose_result, 'sitting')
        if hip_knee_score > 0:
            criteria_met += hip_knee_score
        total_criteria += 1
        
        if total_criteria > 0:
            score = criteria_met / total_criteria
        
        return score
    
    def _classify_lying(self, pose_result: Dict[str, Any], 
                       angles: Dict[str, float]) -> float:
        """Classify lying posture."""
        score = 0.0
        criteria_met = 0
        total_criteria = 0
        
        # Check trunk angle (should be close to horizontal)
        if 'trunk_angle' in angles:
            if angles['trunk_angle'] >= self.thresholds['lying']['trunk_angle_min']:
                criteria_met += 1
            total_criteria += 1
        
        # Check body orientation (shoulders and hips should be at similar height)
        orientation_score = self._check_horizontal_orientation(pose_result)
        if orientation_score > 0:
            criteria_met += orientation_score
        total_criteria += 1
        
        # Check if person is low in the frame
        position_score = self._check_low_position(pose_result)
        if position_score > 0:
            criteria_met += position_score
        total_criteria += 1
        
        if total_criteria > 0:
            score = criteria_met / total_criteria
        
        return score
    
    def _classify_standing(self, pose_result: Dict[str, Any], 
                          angles: Dict[str, float]) -> float:
        """Classify standing posture."""
        score = 0.0
        criteria_met = 0
        total_criteria = 0
        
        # Check leg angles (should be relatively straight)
        if 'left_leg_angle' in angles and 'right_leg_angle' in angles:
            leg_angle = (angles['left_leg_angle'] + angles['right_leg_angle']) / 2
            if leg_angle >= self.thresholds['standing']['leg_angle_min']:
                criteria_met += 1
            total_criteria += 1
        
        # Check trunk angle (should be relatively upright)
        if 'trunk_angle' in angles:
            if angles['trunk_angle'] <= self.thresholds['standing']['trunk_angle_max']:
                criteria_met += 1
            total_criteria += 1
        
        # Check hip angles (should be relatively straight)
        if 'left_hip_angle' in angles and 'right_hip_angle' in angles:
            hip_angle = (angles['left_hip_angle'] + angles['right_hip_angle']) / 2
            if hip_angle >= self.thresholds['standing']['hip_angle_min']:
                criteria_met += 1
            total_criteria += 1
        
        # Check vertical position
        position_score = self._check_vertical_position(pose_result)
        if position_score > 0:
            criteria_met += position_score
        total_criteria += 1
        
        if total_criteria > 0:
            score = criteria_met / total_criteria
        
        return score
    
    def _classify_squatting(self, pose_result: Dict[str, Any], 
                           angles: Dict[str, float]) -> float:
        """Classify squatting posture."""
        score = 0.0
        criteria_met = 0
        total_criteria = 0
        
        # Check hip angles (should be bent)
        if 'left_hip_angle' in angles and 'right_hip_angle' in angles:
            hip_angle = (angles['left_hip_angle'] + angles['right_hip_angle']) / 2
            if hip_angle <= self.thresholds['squatting']['hip_angle_max']:
                criteria_met += 1
            total_criteria += 1
        
        # Check leg angles (should be bent)
        if 'left_leg_angle' in angles and 'right_leg_angle' in angles:
            leg_angle = (angles['left_leg_angle'] + angles['right_leg_angle']) / 2
            if leg_angle <= self.thresholds['squatting']['leg_angle_max']:
                criteria_met += 1
            total_criteria += 1
        
        # Check trunk angle
        if 'trunk_angle' in angles:
            if angles['trunk_angle'] <= self.thresholds['squatting']['trunk_angle_max']:
                criteria_met += 1
            total_criteria += 1
        
        if total_criteria > 0:
            score = criteria_met / total_criteria
        
        return score
    
    def _check_hip_knee_position(self, pose_result: Dict[str, Any], 
                                activity: str) -> float:
        """Check relative position of hips and knees."""
        try:
            landmarks = pose_result['landmarks']
            landmark_names = pose_result['landmark_names']
            
            left_hip_idx = landmark_names.index('LEFT_HIP')
            right_hip_idx = landmark_names.index('RIGHT_HIP')
            left_knee_idx = landmark_names.index('LEFT_KNEE')
            right_knee_idx = landmark_names.index('RIGHT_KNEE')
            
            hip_y = (landmarks[left_hip_idx]['y'] + landmarks[right_hip_idx]['y']) / 2
            knee_y = (landmarks[left_knee_idx]['y'] + landmarks[right_knee_idx]['y']) / 2
            
            if activity == 'sitting':
                # For sitting, hips should be at or above knee level
                return 1.0 if hip_y <= knee_y + 0.1 else 0.0
            
        except (ValueError, IndexError):
            pass
        
        return 0.0
    
    def _check_horizontal_orientation(self, pose_result: Dict[str, Any]) -> float:
        """Check if the body is in horizontal orientation."""
        try:
            landmarks = pose_result['landmarks']
            landmark_names = pose_result['landmark_names']
            
            left_shoulder_idx = landmark_names.index('LEFT_SHOULDER')
            right_shoulder_idx = landmark_names.index('RIGHT_SHOULDER')
            left_hip_idx = landmark_names.index('LEFT_HIP')
            right_hip_idx = landmark_names.index('RIGHT_HIP')
            
            shoulder_y = (landmarks[left_shoulder_idx]['y'] + landmarks[right_shoulder_idx]['y']) / 2
            hip_y = (landmarks[left_hip_idx]['y'] + landmarks[right_hip_idx]['y']) / 2
            
            # Check if shoulders and hips are at similar height
            height_diff = abs(shoulder_y - hip_y)
            return 1.0 if height_diff < 0.2 else 0.0
            
        except (ValueError, IndexError):
            pass
        
        return 0.0
    
    def _check_low_position(self, pose_result: Dict[str, Any]) -> float:
        """Check if the person is in a low position in the frame."""
        try:
            landmarks = pose_result['landmarks']
            landmark_names = pose_result['landmark_names']
            
            # Get average y position of key landmarks
            key_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
            y_positions = []
            
            for landmark_name in key_landmarks:
                idx = landmark_names.index(landmark_name)
                y_positions.append(landmarks[idx]['y'])
            
            avg_y = sum(y_positions) / len(y_positions)
            
            # If average position is in lower part of frame, likely lying
            return 1.0 if avg_y > 0.6 else 0.0
            
        except (ValueError, IndexError):
            pass
        
        return 0.0
    
    def _check_vertical_position(self, pose_result: Dict[str, Any]) -> float:
        """Check if the person is in a vertical standing position."""
        try:
            landmarks = pose_result['landmarks']
            landmark_names = pose_result['landmark_names']
            
            head_idx = landmark_names.index('NOSE')
            ankle_indices = [landmark_names.index('LEFT_ANKLE'), landmark_names.index('RIGHT_ANKLE')]
            
            head_y = landmarks[head_idx]['y']
            ankle_y = min(landmarks[idx]['y'] for idx in ankle_indices)
            
            # Check if there's good vertical separation
            vertical_span = ankle_y - head_y
            return 1.0 if vertical_span > 0.4 else 0.0
            
        except (ValueError, IndexError):
            pass
        
        return 0.0
    
    def get_activity_description(self, activity: str) -> str:
        """Get a human-readable description of the activity."""
        descriptions = {
            'sitting': 'Person is in a sitting position',
            'lying': 'Person is lying down',
            'standing': 'Person is standing upright',
            'squatting': 'Person is in a squatting position',
            'unknown': 'Unable to classify the current activity'
        }
        
        return descriptions.get(activity, 'Unknown activity')