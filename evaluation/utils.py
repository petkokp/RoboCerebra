#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robocerebra_utils.py

Utility functions for RoboCerebra evaluation including data loading,
environment handling, and task processing.
"""

import json
import logging
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from PIL import Image
from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *  # noqa: F403

from config import GenerateConfig, SCENE_MAPPINGS, MOVABLE_OBJECT_LIST

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Local implementations of utility functions to avoid external dependencies
# -----------------------------------------------------------------------------


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Args:
        quat (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img, resize_size):
    """Resize image using PIL."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # img is numpy array (H, W, C)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(
        (resize_size[1], resize_size[0]), resample=Image.Resampling.LANCZOS
    )
    return np.array(pil_img)


# -----------------------------------------------------------------------------


def load_actions(json_path: str) -> Dict[str, List[List[str]]]:
    """Load actions from goal.json, supporting both old and new formats with task_step annotations."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, List[List[str]]] = {}
    for obj_id, relations in data.items():
        processed = []
        for item in relations:
            if isinstance(item, dict) and "state_pair" in item and "task_step" in item:
                triple = item["state_pair"]
                if len(triple) == 2:
                    verb, subj = triple
                    processed.append([verb.lower(), subj])
                elif len(triple) == 3:
                    verb, subj, obj = triple
                    processed.append([verb.lower(), subj, obj])
                else:
                    continue
            elif isinstance(item, list):
                if len(item) == 2:
                    verb, subj = item
                    processed.append([verb.lower(), subj])
                elif len(item) == 3:
                    verb, subj, obj = item
                    processed.append([verb.lower(), subj, obj])
                else:
                    continue
            else:
                continue
        result[obj_id] = processed
    return result


def load_actions_with_steps(
    json_path: str,
) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[int]]]:
    """Load actions and their corresponding task steps from goal.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions_result: Dict[str, List[List[str]]] = {}
    steps_result: Dict[str, List[int]] = {}

    for obj_id, relations in data.items():
        processed_actions = []
        processed_steps = []

        for item in relations:
            if isinstance(item, dict) and "state_pair" in item and "task_step" in item:
                triple = item["state_pair"]
                task_step = item["task_step"]

                if len(triple) == 2:
                    verb, subj = triple
                    processed_actions.append([verb.lower(), subj])
                    processed_steps.append(task_step)
                elif len(triple) == 3:
                    verb, subj, obj = triple
                    processed_actions.append([verb.lower(), subj, obj])
                    processed_steps.append(task_step)
            elif isinstance(item, list):
                if len(item) == 2:
                    verb, subj = item
                    processed_actions.append([verb.lower(), subj])
                    processed_steps.append(len(processed_actions) - 1)
                elif len(item) == 3:
                    verb, subj, obj = item
                    processed_actions.append([verb.lower(), subj, obj])
                    processed_steps.append(len(processed_actions) - 1)

        actions_result[obj_id] = processed_actions
        steps_result[obj_id] = processed_steps

    return actions_result, steps_result


def determine_scene_type(bddl_file: Path) -> str:
    """Determine scene type from BDDL filename."""
    filename = bddl_file.name
    for scene_prefix in SCENE_MAPPINGS.keys():
        if filename.startswith(scene_prefix):
            return scene_prefix

    logger.warning(
        f"Could not determine scene type for {filename}, defaulting to COFFEE_TABLESCENE"
    )
    return "COFFEE_TABLESCENE"


def load_init_state(
    cfg: GenerateConfig, task_type: str, case_name: str, log_file=None
) -> Optional[np.ndarray]:
    """Load initial state for a specific case."""
    if not cfg.use_init_files:
        return None

    init_files_dir = Path(cfg.init_files_root)
    use_ideal_files = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}

    if task_type in use_ideal_files:
        init_dir_name = "Ideal"
        from robocerebra_logging import log_message

        log_message(f"Using Ideal init files for task type: {task_type}", log_file)
    else:
        init_dir_name = task_type.replace(" ", "_")

    init_file = init_files_dir / init_dir_name / f"{case_name}.init"

    if init_file.exists():
        try:
            with open(init_file, "rb") as f:
                init_state = pickle.load(f)
            from robocerebra_logging import log_message

            log_message(f"Loaded init state from {init_file}", log_file)
            return init_state
        except Exception as e:
            logger.error(f"Failed to load init state from {init_file}: {e}")

    logger.warning(f"No init state found for {init_dir_name}/{case_name}")
    return None


def get_task_directories(cfg: GenerateConfig) -> List[Tuple[str, Path]]:
    """Get all task directories for specified task types."""
    robocerebra_root = Path(cfg.robocerebra_root)
    task_dirs = []
    use_ideal_files = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}

    for task_type in cfg.task_types:
        if task_type in use_ideal_files:
            source_dir = robocerebra_root / "Ideal"
            logger.info(f"Using Ideal directory files for task type: {task_type}")
        else:
            source_dir = robocerebra_root / task_type
            logger.info(f"Using original directory files for task type: {task_type}")

        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        for case_dir in source_dir.iterdir():
            if case_dir.is_dir():
                bddl_files = list(case_dir.glob("*.bddl"))
                if bddl_files:
                    task_dirs.append((task_type, case_dir))

    logger.info(f"Found {len(task_dirs)} total task directories")
    return task_dirs


def load_environment(task_dir: Path):
    """Load the environment for a given task directory."""
    bddl_files = list(task_dir.glob("*.bddl"))
    if not bddl_files:
        logger.error(f"No BDDL file found in {task_dir}")
        return None, None, None

    bddl_file = bddl_files[0]

    try:
        problem_info = BDDLUtils.get_problem_info(str(bddl_file))
        problem_name = problem_info["problem_name"]
        scene_type = determine_scene_type(bddl_file)
        expected_class = SCENE_MAPPINGS.get(scene_type, problem_name)
        controller_config = load_controller_config(default_controller="OSC_POSE")

        env = TASK_MAPPING[expected_class](
            bddl_file_name=str(bddl_file),
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        return env, scene_type, str(bddl_file)

    except Exception as e:
        logger.error(f"Failed to load environment for {task_dir}: {e}")
        return None, None, None


def prepare_observation(obs, resize_size):
    """Prepare observation dictionary for model."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image(img, resize_size)
    wrist_img_resized = resize_image(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }
    return observation, img


def process_action(action, model_family=None):
    """
    Deprecated: Action processing should be handled by the model.
    Pass-through for backward compatibility if needed, but models should return
    actions compatible with the environment (e.g. [-1, 1] gripper).
    """
    return action


def _find_obj_y_addr(sim, obj_name: str) -> Optional[int]:
    """Infers the index of the y-coordinate of the object's position in qpos."""
    patterns = [f"{obj_name}_1_joint0", f"{obj_name}_joint0", f"{obj_name}_joint"]
    for jn in patterns:
        if jn in sim.model.joint_names:
            qpos_addr = sim.model.get_joint_qpos_addr(jn)[0]
            return qpos_addr + 1
    return None


def _load_step_objects(json_path: str, step_desc: Sequence[str]) -> List[str]:
    """Parse task_description.json for dynamic distractors."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {item["step"]: item["object"] for item in data if "step" in item}
    objs = []
    for desc in step_desc:
        key = f"Step: {desc}"
        objs.append(mapping.get(key, ""))
    return objs


def parse_task_description(txt_path: str) -> Tuple[List[str], List[int]]:
    """Parse task_description*.txt and return (step_descriptions, start_indices)."""
    step_desc: list[str] = []
    start_indices: list[int] = []
    BRACKET_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        logger.error(f"Error reading task description file {txt_path}: {e}")
        return step_desc, start_indices

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Step"):
            desc = line.split(":", 1)[1].strip()
            step_desc.append(desc)
            if i + 1 < len(lines):
                m = BRACKET_RE.match(lines[i + 1])
                if m:
                    start_idx = int(m.group(1))
                    start_indices.append(start_idx)
                    i += 1
            i += 1
        else:
            i += 1

    return step_desc, start_indices


def setup_dynamic_distractor_info(
    cfg: GenerateConfig, task_dir: Path, env, naming_step_desc: List[str], log_file=None
) -> Optional[Dict[str, Any]]:
    """Setup dynamic distractor information for dynamic object movement."""
    from robocerebra_logging import log_message

    if not (cfg.dynamic and cfg.resume):
        return None

    dir_path = str(task_dir)
    if cfg.task_description_suffix:
        json_name = f"task_description{cfg.task_description_suffix}.json"
    else:
        json_name = "task_description.json"
    json_path = os.path.join(dir_path, json_name)

    if not os.path.isfile(json_path):
        log_message(
            f"[WARN] {json_path} not found, dynamic functionality disabled.", log_file
        )
        return None

    step_objects = _load_step_objects(json_path, naming_step_desc)
    step_addr_y: List[Optional[int]] = []
    step_base_y: List[Optional[float]] = []

    for i, obj_name in enumerate(step_objects):
        if not obj_name or obj_name not in MOVABLE_OBJECT_LIST:
            step_addr_y.append(None)
            step_base_y.append(None)
            continue

        addr = _find_obj_y_addr(env.sim, obj_name)
        if addr is None:
            log_message(
                f"[WARN] Could not find joint for {obj_name}, ignoring the related object.",
                log_file,
            )
            step_addr_y.append(None)
            step_base_y.append(None)
        else:
            step_addr_y.append(addr)
            step_base_y.append(env.sim.data.qpos[addr].copy())

    unrelated_addr = []
    for name in MOVABLE_OBJECT_LIST:
        if name in step_objects:
            continue
        addr = _find_obj_y_addr(env.sim, name)
        if addr is not None:
            unrelated_addr.append((addr, env.sim.data.qpos[addr].copy()))

    if any(a is not None for a in step_addr_y) and unrelated_addr:
        return {
            "step_addr": step_addr_y,
            "step_base": step_base_y,
            "unrel": unrelated_addr,
        }
    else:
        log_message(
            f"[WARN] Insufficient dynamic information, feature disabled.", log_file
        )
        return None
