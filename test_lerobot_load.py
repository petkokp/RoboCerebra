import torch
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from evaluation.config import GenerateConfig
from evaluation.models.lerobot_model import LerobotModel

logging.basicConfig(level=logging.INFO)


def test_load():
    cfg = GenerateConfig(
        model_family="lerobot",
        lerobot_checkpoint="lerobot/pi0_libero_finetuned",
        use_wandb=False,
    )

    model = LerobotModel()
    print("Loading model...")
    model.load(cfg)
    print("Model loaded successfully!")

    # Create dummy observation
    obs = {
        "full_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(7, dtype=np.float32),
    }

    print("Predicting action...")
    action = model.predict_action(
        obs, "pick up the chocolate pudding from wooden cabinet"
    )
    print(f"Action predicted: {action}")
    print(f"Action shape: {action.shape}")


if __name__ == "__main__":
    test_load()
