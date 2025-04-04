"""
Tasks package for the AgentNet Customer Support System.

This package contains the task processing components:
- Task: Task definition and types
- TaskProcessor: Task processing logic
"""

from .task import Task, CustomerSupportTask
from .task_processor import TaskProcessor

__all__ = ["Task", "CustomerSupportTask", "TaskProcessor"] 