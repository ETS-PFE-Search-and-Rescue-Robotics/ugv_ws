"""
PoseSense - Human Pose Detection and Activity Classification Library

A Python library that uses MediaPipe for pose detection and classifies human activities
such as sitting, lying down, standing, and other postures.

Created by Proleak Innovation
Email: center@proleakinnovation.com
Website: https://proleakinnovation.com
"""

from .detector import PoseDetector
from .classifier import ActivityClassifier
from .utils import PoseUtils

__version__ = "1.0.0"
__author__ = "Proleak Innovation"
__email__ = "center@proleakinnovation.com"

__all__ = [
    "PoseDetector",
    "ActivityClassifier", 
    "PoseUtils"
]