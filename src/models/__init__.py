"""PhysDiff-Net model components for synthetic data generation."""

from .encoder import TS2VecWrapper
from .diffusion import ConditionalDiffusion, NoiseScheduler
from .physics import PhysicsConstraints, CapacityLoss, InterferenceLoss, SpatialCorrelationLoss
from .physdiff_net import PhysDiffNet
from .physdiff_net_v4 import PhysDiffNetV4, create_physdiff_v4

# V5 components
from .hour_conditioning import ConcatenationHourConditioning, HybridHourConditioning
from .hour_classifier import AuxiliaryHourClassifier, MultiScaleHourClassifier
from .pcgrad import PCGrad, PCGradOptimizer, GradientMonitor
from .physdiff_net_v5 import PhysDiffNetV5, create_physdiff_v5

__all__ = [
    "TS2VecWrapper",
    "ConditionalDiffusion",
    "NoiseScheduler",
    "PhysicsConstraints",
    "CapacityLoss",
    "InterferenceLoss",
    "SpatialCorrelationLoss",
    "PhysDiffNet",
    "PhysDiffNetV4",
    "create_physdiff_v4",
    # V5 exports
    "ConcatenationHourConditioning",
    "HybridHourConditioning",
    "AuxiliaryHourClassifier",
    "MultiScaleHourClassifier",
    "PCGrad",
    "PCGradOptimizer",
    "GradientMonitor",
    "PhysDiffNetV5",
    "create_physdiff_v5",
]
