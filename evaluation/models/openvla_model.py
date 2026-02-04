import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from evaluation.model_interface import RoboCerebraModel
from evaluation.config import GenerateConfig

logger = logging.getLogger(__name__)


class OpenVLAModel(RoboCerebraModel):
    """
    Stub for OpenVLAModel to handle missing dependencies gracefully.
    OpenVLA-OFT is currently isolated due to TensorFlow dependency conflicts
    with newer Transformers versions required for PI0/SmolVLA.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self._available = False
        self._error_msg = ""

        # Add openvla-oft to path if needed to find 'experiments'
        project_root = Path(__file__).resolve().parent.parent.parent
        openvla_path = project_root / "openvla-oft"
        if str(openvla_path) not in sys.path and openvla_path.exists():
            sys.path.append(str(openvla_path))

        try:
            # Try lazy imports to avoid triggering TF at class init
            import experiments.robot.openvla_utils
            import experiments.robot.robot_utils

            self._available = True
        except ImportError as e:
            self._available = False
            self._error_msg = str(e)
            logger.warning(f"OpenVLAModel dependencies not found: {e}")

    def load(self, cfg: GenerateConfig) -> None:
        if not self._available:
            raise ImportError(
                f"OpenVLAModel is not available because of missing dependencies: {self._error_msg}. "
                "Note: TensorFlow was removed to support newer Transformers versions for PI0/SmolVLA. "
                "OpenVLA functionality is currently disabled."
            )

        from experiments.robot.openvla_utils import (
            get_action_head,
            get_noisy_action_projector,
            get_processor,
            get_proprio_projector,
        )
        from experiments.robot.robot_utils import (
            get_model,
            get_image_resize_size,
        )
        from collections import deque

        self.cfg = cfg
        self.model = get_model(cfg)
        self.proprio_projector = (
            get_proprio_projector(cfg, self.model.llm_dim, proprio_dim=8)
            if cfg.use_proprio
            else None
        )
        self.action_head = (
            get_action_head(cfg, self.model.llm_dim)
            if (cfg.use_l1_regression or cfg.use_diffusion)
            else None
        )
        self.noisy_action_projector = (
            get_noisy_action_projector(cfg, self.model.llm_dim)
            if cfg.use_diffusion
            else None
        )
        self.processor = get_processor(cfg)
        self.resize_size = get_image_resize_size(cfg)

        # Handle unnorm key logic
        unnorm_key = cfg.unnorm_key
        if unnorm_key not in self.model.norm_stats:
            available_keys = list(self.model.norm_stats.keys())
            if f"{unnorm_key}_no_noops" in self.model.norm_stats:
                unnorm_key = f"{unnorm_key}_no_noops"
            elif "bridge_orig" in available_keys:
                unnorm_key = "bridge_orig"
            elif "bridge" in available_keys:
                unnorm_key = "bridge"
            elif len(available_keys) > 0:
                unnorm_key = available_keys[0]
            logger.info(f"Falling back to un-norm key: {unnorm_key}")
        cfg.unnorm_key = unnorm_key

        self.action_queue = deque(maxlen=cfg.num_open_loop_steps)

    def predict_action(self, observation: Dict[str, Any], task_description: str) -> Any:
        from experiments.robot.openvla_utils import get_vla_action
        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )

        if not self.action_queue:
            model_obs = {
                "full_image": observation["full_image"],
                "state": observation["state"],
            }
            if "wrist_image" in observation:
                model_obs["wrist_image"] = observation["wrist_image"]

            actions = get_vla_action(
                cfg=self.cfg,
                vla=self.model,
                processor=self.processor,
                obs=model_obs,
                task_label=task_description,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                use_film=self.cfg.use_film,
            )
            self.action_queue.extend(actions)

        action = self.action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return action

    def reset(self) -> None:
        from collections import deque

        if hasattr(self, "cfg") and self.cfg:
            self.action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
        else:
            self.action_queue = deque()

    def get_image_size(self) -> Union[int, tuple]:
        return getattr(self, "resize_size", 224)
