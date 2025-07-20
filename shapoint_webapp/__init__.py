"""
SHAPoint WebApp - Interactive risk assessment using interpretable machine learning.

This package provides a Streamlit web application for the SHAPoint interpretable
risk assessment framework. It enables users to build, visualize, and deploy
interpretable risk models through an intuitive web interface.
"""

__version__ = "0.1.0"
__author__ = "Tomer D. Meirman"
__email__ = "meirmant@post.bgu.ac.il"

# Import main components for programmatic access
from .model_manager import ModelManager
from .visualizations import RiskVisualizer, UIComponents

__all__ = [
    "ModelManager",
    "RiskVisualizer", 
    "UIComponents",
    "__version__",
    "__author__",
    "__email__"
] 