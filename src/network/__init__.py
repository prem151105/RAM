"""
Network package for the AgentNet Customer Support System.

This package contains the network management components:
- Network: Agent network management
- WeightUpdate: Connection weight management
"""

from .network import Network
from .weight_update import WeightUpdate

__all__ = ["Network", "WeightUpdate"] 