"""
7DMH-QA Crystal Architecture - Engines Package
Manifold-Constrained Holographic Quantum Architecture
"""

from .inference_engine import CrystalInferenceEngine, CrystalStreamingInference
from .training_pipeline import CrystalTrainingPipeline, TrainingConfig, ManifoldLoss
from .optimization_core import SacredBoundsOptimizer, ManifoldCurvatureOptimizer, AutoTuner

__all__ = [
    # Inference
    'CrystalInferenceEngine',
    'CrystalStreamingInference',
    
    # Training
    'CrystalTrainingPipeline',
    'TrainingConfig',
    'ManifoldLoss',
    
    # Optimization
    'SacredBoundsOptimizer',
    'ManifoldCurvatureOptimizer',
    'AutoTuner',
]

__version__ = '2.0.0'
