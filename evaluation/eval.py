#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py

Main evaluation script for RoboCerebra tasks using generic models (OpenVLA, Lerobot, etc.).
"""

import logging
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import draccus
import numpy as np
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import GenerateConfig, validate_config, MOVABLE_OBJECT_LIST
from robocerebra_logging import setup_logging, log_message, save_results_log
from task_runner import (
    setup_task_environment,
    load_task_data,
    setup_task_descriptions,
    validate_task_configuration,
)
from episode import (
    setup_dynamic_distractor_info,
    initialize_episode_state,
    handle_dynamic_movement,
    handle_segment_transition,
    execute_policy_step,
    update_completion_tracking,
    finalize_episode,
    get_libero_dummy_action,
)
from utils import get_task_directories, prepare_observation
from resume import create_step_based_resume_handler
from model_interface import RoboCerebraModel

# --------------------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Model Initialization
# --------------------------------------------------------------------------------------------------


def initialize_model(cfg: GenerateConfig) -> RoboCerebraModel:
    """Initialize model based on configuration."""
    if cfg.model_family == "openvla":
        from models.openvla_model import OpenVLAModel

        model = OpenVLAModel()
    elif cfg.model_family == "lerobot":
        from models.lerobot_model import LerobotModel

        model = LerobotModel()
    else:
        # Try to dynamically import if possible, or raise error
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    model.load(cfg)
    return model


# --------------------------------------------------------------------------------------------------
# Episode and Task Functions
# --------------------------------------------------------------------------------------------------


def run_episode(
    cfg: GenerateConfig,
    env,
    naming_step_desc: Sequence[str],
    model_step_desc: Sequence[str],
    step_states: Sequence[np.ndarray] | None,
    model: RoboCerebraModel,
    goal: Any,
    resize_size,
    log_file=None,
    episode_idx: int = 0,
    distractor_info: Optional[Dict[str, Any]] = None,
    task_line: str | None = None,
    task_name: str = "",
    wait_flag=True,
    task_type: str = "",
    case_name: str = "",
    initial_state: Optional[np.ndarray] = None,
    resume_handler: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, int, int]:
    """Run a single evaluation episode."""

    # Calculate segment count
    segment_count = (
        len(naming_step_desc) if cfg.task_description_suffix else len(model_step_desc)
    )
    full_description = task_line or "" if cfg.complete_description else None

    # Initialize episode state
    obs, episode_stats = initialize_episode_state(
        cfg,
        env,
        goal,
        step_states,
        initial_state,
        task_type,
        case_name,
        resume_handler,
        log_file,
    )

    # Reset model state (e.g. action queue or history)
    model.reset()

    # Initialize dynamic variables
    resume_trigger_step = None
    rng = None
    toggle_dir = -1
    seg_mid_moved = False

    if cfg.dynamic and distractor_info:
        step_addr_y = distractor_info["step_addr"]
        step_base_y = distractor_info["step_base"]
        unrelated_set = distractor_info["unrel"]
        rng = np.random.default_rng()

    # Initialize episode tracking
    seg_increment_accum = 0
    replay_images_all: List[np.ndarray] = []
    replay_images_seg: List[np.ndarray] = []
    t = 0
    max_steps = cfg.switch_steps * segment_count
    prev_step_idx = 0

    # Initial completion baseline
    total_completed_prev = 0
    if not wait_flag:
        comp_start_dict, total_completed_prev, _ = env._check_success(goal)
        if cfg.dynamic_shift_description:
            log_message(
                f"[Dynamic Shift] Final completion baseline: {total_completed_prev} total, details: {comp_start_dict}",
                log_file,
            )
    else:
        # If waiting, we initialize comp_start_dict later
        comp_start_dict = {}

    # Main control loop
    while t < max_steps:
        # Initial waiting period
        if t < cfg.num_steps_wait and wait_flag:
            obs, _, _, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if t == cfg.num_steps_wait and wait_flag:
            comp_start_dict, total_completed_prev, _ = env._check_success(goal)
            if cfg.dynamic_shift_description:
                log_message(
                    f"[Dynamic Shift] Post-wait completion baseline: {total_completed_prev} total, details: {comp_start_dict}",
                    log_file,
                )

        # Calculate segment index
        step_idx = (t // cfg.switch_steps) % segment_count

        # Record segment start
        if t % cfg.switch_steps == 0:
            if cfg.dynamic and distractor_info:
                seg_mid_moved = False

        # Handle segment switching
        if t > 0 and step_idx != prev_step_idx:
            (
                comp_start_dict,
                replay_images_seg,
                seg_increment_accum,
                _,
                skip_increment,
                new_trigger,
            ) = handle_segment_transition(
                cfg,
                env,
                goal,
                step_idx,
                prev_step_idx,
                seg_increment_accum,
                replay_images_seg,
                episode_idx,
                naming_step_desc,
                task_type,
                case_name,
                comp_start_dict,
                step_states,
                episode_stats,
                resume_handler,
                log_file,
            )
            episode_stats["skip_increment"] = skip_increment
            if new_trigger is not None:
                resume_trigger_step = t
                # If we skipped/resumed, we should reset model history too?
                # Usually yes for policies that track history.
                model.reset()

        prev_step_idx = step_idx

        # Handle dynamic movement
        if cfg.dynamic and distractor_info:
            obs, toggle_dir, resume_trigger_step, seg_mid_moved = (
                handle_dynamic_movement(
                    cfg,
                    env,
                    distractor_info,
                    step_idx,
                    resume_trigger_step,
                    t,
                    rng,
                    toggle_dir,
                    seg_mid_moved,
                    log_file,
                )
            )

        # Policy inference & execution
        observation, img = prepare_observation(obs, resize_size)
        # Only accumulate images if video saving is enabled
        if cfg.save_episode_videos:
            replay_images_all.append(img)
        if cfg.save_segment_videos:
            replay_images_seg.append(img)

        # Determine description for model
        if cfg.task_description_suffix != "" and not cfg.complete_description:
            desc = naming_step_desc[step_idx]
        else:
            desc = (
                full_description
                if cfg.complete_description
                else model_step_desc[step_idx]
            )

        # Execute policy step (delegates to model)
        action = execute_policy_step(cfg, model, observation, str(desc))

        # Execute action in environment
        obs, _, _, _ = env.step(action)
        t += 1

        # Update completion tracking
        seg_diff, total_completed_prev = update_completion_tracking(
            env, goal, total_completed_prev, episode_stats, step_idx, log_file
        )
        seg_increment_accum += seg_diff

        if episode_stats["skip_increment"]:
            episode_stats["skip_increment"] = False

    # Finalize episode
    return finalize_episode(
        cfg,
        env,
        goal,
        replay_images_all,
        replay_images_seg,
        episode_idx,
        prev_step_idx,
        seg_increment_accum,
        naming_step_desc,
        task_type,
        case_name,
        episode_stats,
        log_file,
    )


def run_task(
    cfg: GenerateConfig,
    task_type: str,
    task_dir: Path,
    model: RoboCerebraModel,
    resize_size,
    log_file=None,
) -> Tuple[int, int, int, int, Dict]:
    """Evaluate a single task directory."""
    env, bddl_file_path, error = setup_task_environment(task_dir, log_file)
    if error:
        return (
            0,
            0,
            0,
            0,
            {
                "task_type": task_type,
                "case_name": task_dir.name,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0,
                "agent_subtasks": 0,
                "possible_subtasks": 0,
                "subtask_rate": 0,
                "bddl_file": bddl_file_path,
                "used_init_files": cfg.use_init_files,
                "has_step_annotations": False,
                "error": error,
            },
        )

    orig_states, goal, goal_steps, error = load_task_data(task_dir, log_file)
    if error:
        return (
            0,
            0,
            0,
            0,
            {
                "task_type": task_type,
                "case_name": task_dir.name,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0,
                "agent_subtasks": 0,
                "possible_subtasks": 0,
                "subtask_rate": 0,
                "bddl_file": bddl_file_path,
                "used_init_files": cfg.use_init_files,
                "has_step_annotations": bool(goal_steps),
                "error": error,
            },
        )

    naming_step_desc, model_step_desc, start_indices, task_line, error = (
        setup_task_descriptions(cfg, task_dir, log_file)
    )
    if error:
        return (
            0,
            0,
            0,
            0,
            {
                "task_type": task_type,
                "case_name": task_dir.name,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0,
                "agent_subtasks": 0,
                "possible_subtasks": 0,
                "subtask_rate": 0,
                "bddl_file": bddl_file_path,
                "used_init_files": cfg.use_init_files,
                "has_step_annotations": bool(goal_steps),
                "error": error,
            },
        )

    # Validate configuration
    is_valid, base_result = validate_task_configuration(
        cfg,
        naming_step_desc,
        start_indices,
        model_step_desc,
        goal,
        goal_steps,
        bddl_file_path,
        task_type,
        task_dir.name,
        log_file,
    )
    if not is_valid:
        return 0, 0, 0, 0, base_result

    # Setup dynamic distractor info and initial states
    from utils import load_init_state

    distractor_info = setup_dynamic_distractor_info(
        cfg, env, naming_step_desc, str(task_dir), log_file
    )
    initial_states = (
        [load_init_state(cfg, task_type, task_dir.name, log_file)]
        if cfg.use_init_files
        else None
    )
    wait_flag = start_indices[0] == 0
    step_states = [orig_states[idx] for idx in start_indices]
    resume_handler = (
        create_step_based_resume_handler(goal, goal_steps)
        if goal and goal_steps
        else {}
    )

    # Run episodes
    episodes = cfg.num_trials_per_task
    successes = 0
    task_agent_subtasks = 0
    task_possible_subtasks = 0

    for ep_idx in range(episodes):
        log_message(f"  Episode {ep_idx + 1}/{episodes}", log_file)
        initial_state = None
        if initial_states and initial_states[0] is not None:
            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[0]
            else:
                initial_state = initial_states[ep_idx % len(initial_states)]

        succ, ep_subtasks, ep_goals = run_episode(
            cfg,
            env,
            naming_step_desc,
            model_step_desc,
            step_states,
            model,
            goal,
            resize_size,
            log_file,
            episode_idx=ep_idx,
            distractor_info=distractor_info,
            task_line=task_line,
            task_name=task_dir.name,
            wait_flag=wait_flag,
            task_type=task_type,
            case_name=task_dir.name,
            initial_state=initial_state,
            resume_handler=resume_handler,
        )
        successes += int(succ)
        task_agent_subtasks += ep_subtasks
        task_possible_subtasks += ep_goals

    # Clean up
    try:
        env.close()
    except:
        pass

    # Prepare result
    task_result = {
        "task_type": task_type,
        "case_name": task_dir.name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes > 0 else 0,
        "agent_subtasks": task_agent_subtasks,
        "possible_subtasks": task_possible_subtasks,
        "subtask_rate": task_agent_subtasks / task_possible_subtasks
        if task_possible_subtasks > 0
        else 0,
        "bddl_file": bddl_file_path,
        "used_init_files": cfg.use_init_files,
        "has_step_annotations": bool(goal_steps),
        "configuration": {
            "dynamic": cfg.dynamic,
            "dynamic_shift_description": cfg.dynamic_shift_description,
            "resume": cfg.resume,
            "complete_description": cfg.complete_description,
            "excludes_forced_completions": cfg.dynamic_shift_description,
        },
    }

    return episodes, successes, task_agent_subtasks, task_possible_subtasks, task_result


@draccus.wrap()
def eval_robocerebra(cfg: GenerateConfig) -> float:
    """Main evaluation function."""
    validate_config(cfg)

    # Initialize model via generic interface
    model = initialize_model(cfg)

    # Use model's preferred image size
    resize_size = model.get_image_size()

    log_file, _, run_id, results_log_filepath = setup_logging(cfg)

    log_message(f"Starting RoboCerebra evaluation", log_file)
    log_message(f"Model family: {cfg.model_family}", log_file)
    log_message(f"RoboCerebra root: {cfg.robocerebra_root}", log_file)
    log_message(f"Init files root: {cfg.init_files_root}", log_file)
    log_message(f"Use init files: {cfg.use_init_files}", log_file)
    log_message(f"Task types: {cfg.task_types}", log_file)
    log_message(
        f"Dynamic parameters - dynamic: {cfg.dynamic}, dynamic_shift_description: {cfg.dynamic_shift_description}, resume: {cfg.resume}",
        log_file,
    )

    # Get all task directories
    task_dirs = get_task_directories(cfg)

    total_eps = 0
    total_success = 0
    total_agent_subtasks = 0
    total_possible_subtasks = 0
    results_by_task_type = {}
    all_task_results = []

    # Calculate total tasks for progress tracking
    import time
    total_tasks = len(task_dirs)
    completed_tasks = 0
    eval_start_time = time.time()
    task_times = []

    log_message(f"=" * 60, log_file)
    log_message(f"STARTING EVALUATION: {total_tasks} tasks total", log_file)
    log_message(f"=" * 60, log_file)

    # Group tasks by type for better reporting
    for task_type in cfg.task_types:
        task_type_dirs = [(tt, td) for tt, td in task_dirs if tt == task_type]

        if not task_type_dirs:
            log_message(f"No tasks found for task type: {task_type}", log_file)
            continue

        # Configure parameters for each task type individually
        original_dynamic = cfg.dynamic
        original_dynamic_shift = cfg.dynamic_shift_description
        original_resume = cfg.resume

        # Set parameters based on current task type
        if task_type == "Ideal":
            cfg.dynamic = False
            cfg.dynamic_shift_description = False
            cfg.resume = True
        elif task_type == "Mix":
            cfg.dynamic = True
            cfg.dynamic_shift_description = True
            cfg.resume = True
        elif task_type == "Random_Disturbance":
            cfg.dynamic = True
            cfg.dynamic_shift_description = False
            cfg.resume = True
        elif task_type == "Observation_Mismatching":
            cfg.dynamic = False
            cfg.dynamic_shift_description = True
            cfg.resume = True
        else:
            # Other tasks (Memory_Execution, Memory_Exploration): use defaults
            cfg.dynamic = False
            cfg.dynamic_shift_description = False
            cfg.resume = True

        log_message(
            f"Evaluating {len(task_type_dirs)} tasks for task type: {task_type}",
            log_file,
        )
        log_message(
            f"Task type {task_type} - dynamic: {cfg.dynamic}, dynamic_shift_description: {cfg.dynamic_shift_description}, resume: {cfg.resume}",
            log_file,
        )

        task_type_episodes = 0
        task_type_successes = 0
        task_type_agent_subtasks = 0
        task_type_possible_subtasks = 0

        for _, task_dir in task_type_dirs:
            task_start_time = time.time()
            completed_tasks += 1

            # Progress header
            elapsed = time.time() - eval_start_time
            if task_times:
                avg_task_time = sum(task_times) / len(task_times)
                remaining_tasks = total_tasks - completed_tasks
                eta_seconds = remaining_tasks * avg_task_time
                eta_str = f", ETA: {eta_seconds/60:.1f}min" if eta_seconds > 60 else f", ETA: {eta_seconds:.0f}s"
            else:
                eta_str = ""

            log_message(
                f"\n{'='*60}\n"
                f"[PROGRESS] Task {completed_tasks}/{total_tasks} ({100*completed_tasks/total_tasks:.0f}%) | "
                f"Type: {task_type} | Case: {task_dir.name} | Elapsed: {elapsed/60:.1f}min{eta_str}\n"
                f"{'='*60}",
                log_file,
            )

            eps, succ, subtasks, possible, task_result = run_task(
                cfg,
                task_type,
                task_dir,
                model,
                resize_size,
                log_file,
            )

            # Track task time for ETA calculation
            task_time = time.time() - task_start_time
            task_times.append(task_time)
            log_message(f"[PROGRESS] Task completed in {task_time:.1f}s", log_file)

            all_task_results.append(task_result)
            task_type_episodes += eps
            task_type_successes += succ
            task_type_agent_subtasks += subtasks
            task_type_possible_subtasks += possible
            total_eps += eps
            total_success += succ
            total_agent_subtasks += subtasks
            total_possible_subtasks += possible

        # Log task type results
        task_type_success_rate = (
            task_type_successes / task_type_episodes if task_type_episodes > 0 else 0
        )
        task_type_subtask_rate = (
            task_type_agent_subtasks / task_type_possible_subtasks
            if task_type_possible_subtasks > 0
            else 0
        )

        results_by_task_type[task_type] = {
            "episodes": task_type_episodes,
            "successes": task_type_successes,
            "success_rate": task_type_success_rate,
            "subtask_rate": task_type_subtask_rate,
            "agent_subtasks": task_type_agent_subtasks,
            "possible_subtasks": task_type_possible_subtasks,
        }

        log_message(
            f"Task type {task_type} complete: "
            f"Episode success rate: {task_type_success_rate:.2%} ({task_type_successes}/{task_type_episodes}), "
            f"Subtask success rate: {task_type_subtask_rate:.2%} ({task_type_agent_subtasks}/{task_type_possible_subtasks})",
            log_file,
        )

        # Restore original parameters
        cfg.dynamic = original_dynamic
        cfg.dynamic_shift_description = original_dynamic_shift
        cfg.resume = original_resume

    # Final results
    overall_success_rate = total_success / total_eps if total_eps > 0 else 0
    overall_subtask_rate = (
        total_agent_subtasks / total_possible_subtasks
        if total_possible_subtasks > 0
        else 0
    )

    total_eval_time = time.time() - eval_start_time
    log_message("\n" + "=" * 60, log_file)
    log_message("EVALUATION COMPLETE", log_file)
    log_message(f"Total time: {total_eval_time/60:.1f} minutes ({total_eval_time:.0f}s)", log_file)
    log_message(f"Tasks evaluated: {completed_tasks}", log_file)
    if task_times:
        log_message(f"Average time per task: {sum(task_times)/len(task_times):.1f}s", log_file)
    log_message("=" * 60, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 60, log_file)

    for task_type, results in results_by_task_type.items():
        log_message(
            f"{task_type}: Episode {results['success_rate']:.2%} ({results['successes']}/{results['episodes']}), "
            f"Subtask {results['subtask_rate']:.2%} ({results['agent_subtasks']}/{results['possible_subtasks']})",
            log_file,
        )

    log_message(
        f"OVERALL: Episode {overall_success_rate:.2%} ({total_success}/{total_eps}), "
        f"Subtask {overall_subtask_rate:.2%} ({total_agent_subtasks}/{total_possible_subtasks})",
        log_file,
    )

    # Save detailed results to JSON log
    save_results_log(
        results_log_filepath,
        cfg,
        results_by_task_type,
        total_eps,
        total_success,
        total_agent_subtasks,
        total_possible_subtasks,
        run_id,
        all_task_results,
    )

    # Log to wandb if enabled
    if cfg.use_wandb:
        for task_type, results in results_by_task_type.items():
            wandb.log(
                {
                    f"success_rate/{task_type}": results["success_rate"],
                    f"subtask_rate/{task_type}": results["subtask_rate"],
                    f"num_episodes/{task_type}": results["episodes"],
                }
            )
        wandb.log(
            {
                "success_rate/overall": overall_success_rate,
                "subtask_rate/overall": overall_subtask_rate,
                "num_episodes/total": total_eps,
            }
        )

    # Close log file
    if log_file:
        log_file.close()

    return overall_success_rate


if __name__ == "__main__":
    eval_robocerebra()
