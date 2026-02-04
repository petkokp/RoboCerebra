from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import torch


class RoboCerebraModel(ABC):
    """Abstract base class for models evaluated in RoboCerebra."""

    @abstractmethod
    def load(self, cfg: Any) -> None:
        """
        Load the model weights and necessary components (processors, etc.).

        Args:
            cfg: Configuration object containing model parameters.
        """
        pass

    @abstractmethod
    def predict_action(
        self, observation: Dict[str, Any], task_description: str
    ) -> np.ndarray:
        """
        Predict an action based on the observation and task description.

        Args:
            observation: Dictionary containing:
                - 'full_image': np.ndarray (H, W, 3) - Primary camera image
                - 'wrist_image': np.ndarray (H, W, 3) - Wrist camera image
                - 'state': np.ndarray - Proprioceptive state
            task_description: String description of the current task step.

        Returns:
            np.ndarray: The predicted action vector (typically 7-dim: x, y, z, r, p, y, gripper).
                        The gripper action should be normalized to [-1, 1] if applicable.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the model's internal state.
        Called at the beginning of each episode.
        Useful for recurrent policies or diffusion policies with history.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Union[int, tuple]:
        """
        Get the expected input image size for this model.

        Returns:
            Union[int, tuple]: (height, width) or int if square.
        """
        pass
