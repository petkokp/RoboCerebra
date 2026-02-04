import torch
import numpy as np
import sys
import logging
import types
import json
import importlib.util
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
from packaging import version

from evaluation.model_interface import RoboCerebraModel
from evaluation.config import GenerateConfig

logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================


def check_dependencies():
    """Verify dependencies."""
    # Check Transformers version for PaliGemma/PI0 support
    try:
        import transformers

        current_version = version.parse(transformers.__version__)
        required_version = version.parse("4.41.0")

        if current_version < required_version:
            logger.warning(
                f"Transformers version {current_version} is detected. "
                f"PI0 and PaliGemma-based models require transformers >= 4.41.0. "
            )
    except ImportError:
        logger.warning("Transformers not found!")


check_dependencies()

# =============================================================================
# MONKEY PATCHES
# =============================================================================


def apply_pi0_patches():
    """Apply fixes for PI0 model issues in specific torch/lerobot versions."""
    # 1. Disable torch compilation globally before any model logic
    try:
        import torch._dynamo

        torch._dynamo.disable()
        torch.compile = lambda x, **kwargs: x
        logger.info("Disabled torch.compile and Dynamo globally")
    except Exception as e:
        logger.debug(f"Failed to disable Dynamo: {e}")

    # 2. Transformers check patch (bypass strict version check in LeRobot)
    try:
        import transformers.models.siglip

        if not hasattr(transformers.models.siglip, "check"):
            m = types.ModuleType("transformers.models.siglip.check")
            m.check_whether_transformers_replace_is_installed_correctly = lambda: True
            sys.modules["transformers.models.siglip.check"] = m
            transformers.models.siglip.check = m
            logger.info("Patched transformers.models.siglip.check for compatibility")
    except ImportError:
        pass

    # 3. PI0 Patch
    try:
        import lerobot.policies.pi0.modeling_pi0 as modeling_pi0

        # Patch PI0Pytorch._prepare_attention_masks_4d
        def patched_prepare(self, att_2d_masks):
            # Ensure att_2d_masks is boolean before passing to torch.where
            if torch.is_tensor(att_2d_masks) and att_2d_masks.dtype != torch.bool:
                att_2d_masks = att_2d_masks.to(torch.bool)

            # Direct implementation
            att_2d_masks_4d = att_2d_masks[:, None, :, :]
            mask_val = getattr(
                modeling_pi0, "OPENPI_ATTENTION_MASK_VALUE", -2.3819763e38
            )
            # Ensure condition is boolean for torch.where
            cond = att_2d_masks_4d.to(torch.bool)
            return torch.where(cond, 0.0, mask_val)

        modeling_pi0.PI0Pytorch._prepare_attention_masks_4d = patched_prepare
        logger.info("Applied torch.where fix to PI0Pytorch")

        # Patch PI0Pytorch.sample_noise to use correct dtype (avoid Float/BFloat16 mismatch)
        def patched_sample_noise(self, shape, device):
            # Use the model's precision instead of hardcoded float32
            target_dtype = next(self.parameters()).dtype
            return torch.normal(
                mean=0.0,
                std=1.0,
                size=shape,
                dtype=target_dtype,
                device=device,
            )

        modeling_pi0.PI0Pytorch.sample_noise = patched_sample_noise
        logger.info("Applied dtype fix to PI0Pytorch.sample_noise")

        # Patch 3: Fix sample_time to use correct dtype
        def patched_sample_time(self, bsize, device):
            from lerobot.policies.pi0.modeling_pi0 import sample_beta

            time_beta = sample_beta(
                self.config.time_sampling_beta_alpha,
                self.config.time_sampling_beta_beta,
                bsize,
                device,
            )
            time = (
                time_beta * self.config.time_sampling_scale
                + self.config.time_sampling_offset
            )
            # CRITICAL: We must ensure the returned time tensor matches the model's precision
            target_dtype = next(self.parameters()).dtype
            # logging.info(f"DEBUG: sample_time using dtype={target_dtype}")
            return time.to(dtype=target_dtype, device=device)

        modeling_pi0.PI0Pytorch.sample_time = patched_sample_time
        logger.info("Applied dtype fix to PI0Pytorch.sample_time")

        # Patch 4: Fix gradient_checkpointing_enable to handle new transformers structure
        def patched_gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True
            # New structure: paligemma.model.language_model
            pg = self.paligemma_with_expert.paligemma
            if hasattr(pg, "model"):
                pg.model.language_model.gradient_checkpointing = True
                pg.model.vision_tower.gradient_checkpointing = True
            else:
                # Fallback for old structure
                pg.language_model.gradient_checkpointing = True
                pg.vision_tower.gradient_checkpointing = True
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing for PI0Pytorch model (patched)")

        modeling_pi0.PI0Pytorch.gradient_checkpointing_enable = (
            patched_gradient_checkpointing_enable
        )

        # Patch 4: Fix BaseModelOutputWithPooling error in embed_prefix
        # In newer transformers, get_image_features returns a dict-like object
        # but LeRobot expects a raw tensor.
        def patched_embed_image(self, image: torch.Tensor):
            features = self.paligemma.model.get_image_features(image)
            if hasattr(features, "pooler_output"):
                return features.pooler_output
            return features

        modeling_pi0.PaliGemmaWithExpertModel.embed_image = patched_embed_image
        logger.info("Applied fix to PaliGemmaWithExpertModel.embed_image")

        # Patch 5: Fix embed_language_tokens for new transformers structure
        def patched_embed_language_tokens(self, tokens: torch.Tensor):
            pg = self.paligemma
            if hasattr(pg, "model"):
                return pg.model.language_model.embed_tokens(tokens)
            return pg.language_model.embed_tokens(tokens)

        modeling_pi0.PaliGemmaWithExpertModel.embed_language_tokens = (
            patched_embed_language_tokens
        )
        logger.info("Applied fix to PaliGemmaWithExpertModel.embed_language_tokens")

        # Patch 6: Fix sample_actions for new transformers structure
        # (AttributeError: 'PaliGemmaForConditionalGeneration' object has no attribute 'language_model')
        _original_sample_actions = modeling_pi0.PI0Pytorch.sample_actions

        def patched_sample_actions(self, *args, **kwargs):
            # Also ensure weights are consistently bfloat16 here just in case
            if any(p.dtype == torch.bfloat16 for p in self.parameters()):
                self.to(torch.bfloat16)

            pg = self.paligemma_with_expert.paligemma
            has_attached = False
            if hasattr(pg, "model") and not hasattr(pg, "language_model"):
                pg.language_model = pg.model.language_model
                has_attached = True

            try:
                return _original_sample_actions(self, *args, **kwargs)
            finally:
                if has_attached and hasattr(pg, "language_model"):
                    delattr(pg, "language_model")

        modeling_pi0.PI0Pytorch.sample_actions = patched_sample_actions
        logger.info("Applied fix to PI0Pytorch.sample_actions")

        # Patch 7: Fix denoise_step to ensure submodule consistency and cast inputs
        original_denoise_step = modeling_pi0.PI0Pytorch.denoise_step

        def patched_denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            # Determine target dtype
            target_dtype = next(self.parameters()).dtype

            # Cast inputs to target dtype
            if torch.is_tensor(state) and state.is_floating_point():
                state = state.to(target_dtype)
            if torch.is_tensor(x_t) and x_t.is_floating_point():
                x_t = x_t.to(target_dtype)
            if torch.is_tensor(timestep) and timestep.is_floating_point():
                timestep = timestep.to(target_dtype)

            # NOTE: Submodule dtype conversion is done ONCE during load(), not per-step

            # Call original denoise_step to compute embeddings
            from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

            # FIX: Check if past_key_values cache size differs from our attention mask size
            # and adjust the attention mask accordingly
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                cached_len = past_key_values.get_seq_length()
                mask_kv_len = full_att_2d_masks_4d.shape[-1]  # prefix_len + suffix_len
                expected_kv_len = cached_len + suffix_len

                if mask_kv_len != expected_kv_len:
                    # Need to pad the attention mask to match cache size
                    # The mask has shape [batch, 1, suffix_len, prefix_len + suffix_len]
                    # We need shape [batch, 1, suffix_len, cached_len + suffix_len]
                    pad_size = expected_kv_len - mask_kv_len
                    if pad_size > 0:
                        # Pad with True (attend) values at the beginning to account for extra cached tokens
                        padding = torch.ones(
                            batch_size, 1, suffix_len, pad_size,
                            dtype=full_att_2d_masks_4d.dtype,
                            device=full_att_2d_masks_4d.device
                        )
                        # Insert padding at the start of the key dimension
                        full_att_2d_masks_4d = torch.cat([padding, full_att_2d_masks_4d], dim=-1)
                        logger.debug(f"Padded attention mask from {mask_kv_len} to {expected_kv_len}")

            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            return self.action_out_proj(suffix_out)

        modeling_pi0.PI0Pytorch.denoise_step = patched_denoise_step
        logger.info("Applied fix to PI0Pytorch.denoise_step for dtype consistency")

        # Patch 8: Fix forward to ensure input casting
        original_select_action = modeling_pi0.PI0Policy.select_action

        def patched_select_action(self, batch, **kwargs):
            # Determine target dtype
            target_dtype = next(self.parameters()).dtype

            # Recursively cast tensors in batch, args and kwargs to target_dtype
            def cast_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.is_floating_point():
                        return obj.to(target_dtype)
                    return obj
                elif isinstance(obj, dict):
                    return {k: cast_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [cast_tensors(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(cast_tensors(v) for v in obj)
                return obj

            batch = cast_tensors(batch)
            kwargs = cast_tensors(kwargs)

            # NOTE: Model dtype conversion is done ONCE during load(), not per-inference
            # The previous code that called self.model.to() and iterated modules here
            # was extremely expensive and caused ~60% GPU idle time

            return original_select_action(self, batch, **kwargs)

        modeling_pi0.PI0Policy.select_action = patched_select_action

        logger.info("Applied fix to PI0Policy.select_action for dtype consistency")

        # NOTE: Previous patch 8 (patched_pg_forward) that called self.gemma_expert.to(target_dtype)
        # on every forward pass has been removed - dtype conversion is done ONCE during load()
        logger.info(
            "Skipping PaliGemmaWithExpertModel.forward patch - dtype handled at load time"
        )

    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not apply PI0 patches: {e}")

    # 3. PI05 Patch
    try:
        import lerobot.policies.pi05.modeling_pi05 as modeling_pi05

        def patched_prepare_pi05(self, att_2d_masks):
            if torch.is_tensor(att_2d_masks) and att_2d_masks.dtype != torch.bool:
                att_2d_masks = att_2d_masks.to(torch.bool)
            att_2d_masks_4d = att_2d_masks[:, None, :, :]
            mask_val = getattr(
                modeling_pi05, "OPENPI_ATTENTION_MASK_VALUE", -2.3819763e38
            )
            cond = att_2d_masks_4d.to(torch.bool)
            return torch.where(cond, 0.0, mask_val)

        modeling_pi05.PI05Pytorch._prepare_attention_masks_4d = patched_prepare_pi05
        logger.info("Applied torch.where fix to PI05Pytorch")

        # Patch PI05 denoise_step for attention mask size fix (similar to PI0)
        original_pi05_denoise_step = modeling_pi05.PI05Pytorch.denoise_step

        def patched_pi05_denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
            """Patched denoise_step that fixes attention mask size mismatch with cache."""
            from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

            # Determine target dtype
            target_dtype = next(self.parameters()).dtype

            # Cast inputs to target dtype
            if torch.is_tensor(x_t) and x_t.is_floating_point():
                x_t = x_t.to(target_dtype)
            if torch.is_tensor(timestep) and timestep.is_floating_point():
                timestep = timestep.to(target_dtype)

            # NOTE: Submodule dtype conversion is done ONCE during load(), not per-step

            # Compute suffix embeddings (PI05 doesn't use state in embed_suffix)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(x_t, timestep)
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

            # FIX: Check if past_key_values cache size differs from our attention mask size
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                cached_len = past_key_values.get_seq_length()
                mask_kv_len = full_att_2d_masks_4d.shape[-1]
                expected_kv_len = cached_len + suffix_len

                if mask_kv_len != expected_kv_len:
                    pad_size = expected_kv_len - mask_kv_len
                    if pad_size > 0:
                        padding = torch.ones(
                            batch_size, 1, suffix_len, pad_size,
                            dtype=full_att_2d_masks_4d.dtype,
                            device=full_att_2d_masks_4d.device
                        )
                        full_att_2d_masks_4d = torch.cat([padding, full_att_2d_masks_4d], dim=-1)
                        logger.debug(f"PI05: Padded attention mask from {mask_kv_len} to {expected_kv_len}")

            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            return self.action_out_proj(suffix_out)

        modeling_pi05.PI05Pytorch.denoise_step = patched_pi05_denoise_step
        logger.info("Applied fix to PI05Pytorch.denoise_step for attention mask consistency")

        # Patch PI05Policy.select_action for dtype consistency
        original_pi05_select_action = modeling_pi05.PI05Policy.select_action

        def patched_pi05_select_action(self, batch, **kwargs):
            target_dtype = next(self.parameters()).dtype

            def cast_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.is_floating_point():
                        return obj.to(target_dtype)
                    return obj
                elif isinstance(obj, dict):
                    return {k: cast_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [cast_tensors(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(cast_tensors(v) for v in obj)
                return obj

            batch = cast_tensors(batch)
            kwargs = cast_tensors(kwargs)

            # NOTE: Model dtype conversion is done ONCE during load(), not per-inference

            return original_pi05_select_action(self, batch, **kwargs)

        modeling_pi05.PI05Policy.select_action = patched_pi05_select_action
        logger.info("Applied fix to PI05Policy.select_action for dtype consistency")

    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not apply PI05 patches: {e}")

    # 4. PI0Fast Patches
    try:
        import lerobot.policies.pi0_fast.modeling_pi0_fast as modeling_pi0_fast

        # Patch to gracefully handle invalid action tokens (return zero actions instead of crashing)
        original_detokenize_actions = modeling_pi0_fast.PI0FastPolicy.detokenize_actions

        def patched_detokenize_actions(self, tokens, action_horizon, action_dim):
            """Patched detokenize_actions that gracefully handles invalid tokens."""
            bsize = tokens.shape[0]

            # Check if tokens start with "Action: " (required by PI0Fast)
            for b in range(bsize):
                token_ids = tokens[b].tolist()
                token_strs = self._paligemma_tokenizer.convert_ids_to_tokens(token_ids)
                if not (len(token_strs) >= 2 and token_strs[0] == "Action" and token_strs[1] == ":"):
                    # Return zero actions as fallback when model doesn't generate proper action tokens
                    if not hasattr(self, "_warned_invalid_tokens"):
                        logger.warning(
                            "PI0Fast: Model is not generating valid action tokens. "
                            "This may indicate a model training issue or input format mismatch. "
                            "Returning zero actions as fallback."
                        )
                        self._warned_invalid_tokens = True
                    return torch.zeros(bsize, action_horizon, action_dim, device=tokens.device)

            # Call original method with validation disabled
            original_validate = self.config.validate_action_token_prefix
            try:
                self.config.validate_action_token_prefix = False
                return original_detokenize_actions(self, tokens, action_horizon, action_dim)
            finally:
                self.config.validate_action_token_prefix = original_validate

        modeling_pi0_fast.PI0FastPolicy.detokenize_actions = patched_detokenize_actions
        logger.info("Applied patch to PI0FastPolicy.detokenize_actions for error handling")

        # Patch dtype consistency for PI0Fast select_action
        original_pi0fast_select_action = modeling_pi0_fast.PI0FastPolicy.select_action

        def patched_pi0fast_select_action(self, batch, **kwargs):
            target_dtype = next(self.parameters()).dtype

            def cast_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.is_floating_point():
                        return obj.to(target_dtype)
                    return obj
                elif isinstance(obj, dict):
                    return {k: cast_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [cast_tensors(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(cast_tensors(v) for v in obj)
                return obj

            batch = cast_tensors(batch)
            kwargs = cast_tensors(kwargs)

            return original_pi0fast_select_action(self, batch, **kwargs)

        modeling_pi0_fast.PI0FastPolicy.select_action = patched_pi0fast_select_action
        logger.info("Applied fix to PI0FastPolicy.select_action for dtype consistency")

    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not apply PI0Fast patches: {e}")


apply_pi0_patches()

# =============================================================================
# LEROBOT IMPORTS
# =============================================================================

from lerobot.policies.factory import get_policy_class
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME


class PolicyAdapter:
    """
    Adapts a LeRobot policy to the RoboCerebra interface.
    """

    def __init__(self, policy, device: str):
        self.policy = policy
        self.device = device
        self.image_keys = self._infer_image_keys()

    def _infer_image_keys(self) -> Dict[str, str]:
        """
        Map RoboCerebra observation keys ('full_image', 'wrist_image')
        to LeRobot policy input keys.
        """
        mapping = {}
        if not hasattr(self.policy.config, "input_features"):
            return mapping

        features = self.policy.config.input_features

        # 1. Wrist image mapping
        wrist_candidates = [k for k in features if "image" in k and "wrist" in k]
        if wrist_candidates:
            mapping["wrist_image"] = wrist_candidates[0]

        # 2. Full image mapping (agent, phone, main, front, high)
        full_candidates = [
            k
            for k in features
            if "image" in k
            and any(x in k for x in ["agent", "phone", "main", "front", "high"])
        ]
        if full_candidates:
            mapping["full_image"] = full_candidates[0]

        # 3. Fallback: if 'full_image' not mapped, take the first available image
        # that isn't already used for wrist
        if "full_image" not in mapping:
            used_values = mapping.values()
            other_images = [
                k for k in features if "image" in k and k not in used_values
            ]
            if other_images:
                mapping["full_image"] = other_images[0]

        logger.info(f"Inferred image mapping: {mapping}")
        return mapping

    def prepare_inputs(
        self, observation: Dict[str, Any], task_description: str
    ) -> Dict[str, torch.Tensor]:
        """Convert RoboCerebra observation dict to LeRobot input dict."""
        inputs = {}

        # Determine target dtype from policy parameters
        try:
            target_dtype = next(self.policy.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        # Process images
        for obs_key, lerobot_key in self.image_keys.items():
            if obs_key in observation:
                img = observation[obs_key]  # H, W, C, uint8
                # Normalize to [0, 1] and CHW format
                # LeRobot expects (B, C, H, W) float32 [0,1]
                img_tensor = (
                    torch.from_numpy(img).to(self.device).to(target_dtype) / 255.0
                )
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                inputs[lerobot_key] = img_tensor

        # Process state if required
        if (
            hasattr(self.policy.config, "input_features")
            and "observation.state" in self.policy.config.input_features
        ):
            if "state" in observation:
                state = (
                    torch.from_numpy(observation["state"])
                    .to(self.device)
                    .to(target_dtype)
                    .unsqueeze(0)
                )
                inputs["observation.state"] = state

        return inputs

    def post_process_action(self, action: torch.Tensor) -> np.ndarray:
        """Convert policy output tensor to numpy array (D,)."""
        # Convert to float32 for numpy compatibility (bfloat16 not supported)
        action_np = action.detach().cpu().float().numpy().squeeze()

        # Handle dimension mismatch
        # RoboCerebra environment expects 7 dim (6 joints/pose + 1 gripper).

        if action_np.ndim > 1:
            # Flatten if somehow we got (1, D)
            action_np = action_np.flatten()

        action_dim = action_np.shape[0]

        if action_dim == 6:
            # Assume 6DoF (joints/pose) without gripper -> append open gripper (-1.0)
            action_np = np.concatenate([action_np, [-1.0]])

        elif action_dim == 7:
            # Already correct dimension
            pass

        elif action_dim == 14:
            # Bimanual output (14 dim) - use first 7 dims
            if not hasattr(self, "_warned_bimanual"):
                logger.warning(
                    "Policy output is 14-dim (Bimanual?). Using first 7 dimensions."
                )
                self._warned_bimanual = True
            action_np = action_np[:7]

        elif action_dim > 7:
            # Model outputs more dimensions than needed (e.g., PI05 base outputs 32)
            # Extract only the first 7 dimensions for the environment
            if not hasattr(self, "_warned_action_dim"):
                logger.warning(
                    f"Policy output is {action_dim}-dim. Using first 7 dimensions for RoboCerebra environment."
                )
                self._warned_action_dim = True
            action_np = action_np[:7]

        return action_np

    def reset(self):
        self.policy.reset()


class VLAAdapter(PolicyAdapter):
    """Adapter for VLA models that require text tokenization."""

    def __init__(self, policy, device: str):
        super().__init__(policy, device)
        self.tokenizer = self._find_tokenizer()
        if not self.tokenizer:
            logger.warning(
                "Tokenizer not found for VLA model! Language inputs will be missing."
            )

    def _find_tokenizer(self):
        """Find the tokenizer within the policy structure."""
        # Standard location
        if hasattr(self.policy, "tokenizer") and self.policy.tokenizer is not None:
            return self.policy.tokenizer

        # Nested model location
        if hasattr(self.policy, "model"):
            if hasattr(self.policy.model, "processor") and hasattr(
                self.policy.model.processor, "tokenizer"
            ):
                return self.policy.model.processor.tokenizer

        return None

    def prepare_inputs(
        self, observation: Dict[str, Any], task_description: str
    ) -> Dict[str, torch.Tensor]:
        inputs = super().prepare_inputs(observation, task_description)

        if self.tokenizer:
            tokens = self.tokenizer(
                [task_description], return_tensors="pt", padding=True, truncation=True
            )
            inputs["observation.language.tokens"] = tokens["input_ids"].to(self.device)
            inputs["observation.language.attention_mask"] = tokens["attention_mask"].to(
                self.device
            )

        return inputs


class PI0Adapter(VLAAdapter):
    """Adapter for PI0 and PI05 models."""

    def __init__(self, policy, device: str):
        super().__init__(policy, device)
        # Get tokenizer max length from policy config (default 48 for PI0)
        self.tokenizer_max_length = getattr(policy.config, "tokenizer_max_length", 48)

    def _find_tokenizer(self):
        # Try standard find first
        tokenizer = super()._find_tokenizer()
        if tokenizer:
            return tokenizer

        # PI0/PI05 specific locations
        if hasattr(self.policy, "model"):
            # PI05 specific path
            if hasattr(self.policy.model, "paligemma_with_expert") and hasattr(
                self.policy.model.paligemma_with_expert, "processor"
            ):
                return self.policy.model.paligemma_with_expert.processor.tokenizer

        # Fallback to loading PaliGemma tokenizer (correct tokenizer for PI0)
        try:
            from transformers import AutoTokenizer

            # PI0 uses PaliGemma tokenizer
            model_id = "google/paligemma-3b-pt-224"
            logger.info(f"Loading fallback tokenizer for PI0 from {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load fallback tokenizer for PI0: {e}")
            return None

    def prepare_inputs(
        self, observation: Dict[str, Any], task_description: str
    ) -> Dict[str, torch.Tensor]:
        """Convert RoboCerebra observation to PI0 input format with proper tokenization."""
        inputs = {}

        # Determine target dtype from policy parameters
        try:
            target_dtype = next(self.policy.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        # Process images - PI0 expects specific image keys
        for obs_key, lerobot_key in self.image_keys.items():
            if obs_key in observation:
                img = observation[obs_key]  # H, W, C, uint8
                # Normalize to [0, 1] and CHW format
                img_tensor = (
                    torch.from_numpy(img).to(self.device).to(target_dtype) / 255.0
                )
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                inputs[lerobot_key] = img_tensor

        # Process state if required
        if (
            hasattr(self.policy.config, "input_features")
            and "observation.state" in self.policy.config.input_features
        ):
            if "state" in observation:
                state = (
                    torch.from_numpy(observation["state"])
                    .to(self.device)
                    .to(target_dtype)
                    .unsqueeze(0)
                )
                inputs["observation.state"] = state

        # Tokenize task description with correct PI0 settings
        if self.tokenizer:
            # PI0 expects a newline at the end of the task description
            if not task_description.endswith("\n"):
                task_description = f"{task_description}\n"

            tokens = self.tokenizer(
                [task_description],
                return_tensors="pt",
                max_length=self.tokenizer_max_length,
                padding="max_length",
                truncation=True,
            )
            inputs["observation.language.tokens"] = tokens["input_ids"].to(self.device)
            inputs["observation.language.attention_mask"] = tokens["attention_mask"].to(
                self.device, dtype=torch.bool
            )

        return inputs


class SmolVLAAdapter(VLAAdapter):
    """Adapter for SmolVLA."""

    def _find_tokenizer(self):
        tokenizer = super()._find_tokenizer()
        if tokenizer:
            return tokenizer

        # SmolVLA specific
        if hasattr(self.policy, "model") and hasattr(
            self.policy.model, "vlm_with_expert"
        ):
            if hasattr(self.policy.model.vlm_with_expert, "processor"):
                return self.policy.model.vlm_with_expert.processor.tokenizer
        return None


class XVLAAdapter(VLAAdapter):
    """
    Adapter for XVLA models.

    XVLA uses Florence2 as the backbone and expects:
    - BART tokenizer (facebook/bart-large by default)
    - Input keys: observation.language.tokens, images, observation.state, domain_id
    """

    def __init__(self, policy, device: str):
        # Get tokenizer settings from policy config BEFORE calling super().__init__
        # because _find_tokenizer is called during parent init
        config_max_length = getattr(policy.config, "tokenizer_max_length", 64)
        max_len_seq = getattr(policy.config, "max_len_seq", 512)

        # IMPORTANT: Ensure tokenizer_max_length fits within max_len_seq
        # The pretrained xvla-base has a config bug where tokenizer_max_length=1024
        # but max_len_seq=512. We cap it to a reasonable value.
        # Reserve space for: soft_prompts (32) + visual tokens (~50) + proprio + action chunk (~30)
        reasonable_max = min(config_max_length, max_len_seq // 4)  # ~128 for max_len_seq=512
        if config_max_length > max_len_seq:
            logger.warning(
                f"XVLA: tokenizer_max_length ({config_max_length}) > max_len_seq ({max_len_seq}). "
                f"Capping to {reasonable_max} to avoid sequence length errors."
            )
        self.tokenizer_max_length = reasonable_max

        self.tokenizer_padding_side = getattr(policy.config, "tokenizer_padding_side", "right")
        self.pad_language_to = getattr(policy.config, "pad_language_to", "max_length")
        self.max_state_dim = getattr(policy.config, "max_state_dim", 32)
        super().__init__(policy, device)

    def _find_tokenizer(self):
        # Try standard find first
        tokenizer = super()._find_tokenizer()
        if tokenizer:
            tokenizer.padding_side = self.tokenizer_padding_side
            return tokenizer

        # XVLA uses BART tokenizer by default
        try:
            from transformers import AutoTokenizer

            tokenizer_name = getattr(self.policy.config, "tokenizer_name", "facebook/bart-large")
            logger.info(f"Loading tokenizer for XVLA from {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer.padding_side = self.tokenizer_padding_side
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer for XVLA: {e}")
            return None

    def prepare_inputs(
        self, observation: Dict[str, Any], task_description: str
    ) -> Dict[str, torch.Tensor]:
        """Convert RoboCerebra observation to XVLA input format."""
        inputs = {}

        # Determine target dtype from policy parameters
        try:
            target_dtype = next(self.policy.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        # Process images
        for obs_key, lerobot_key in self.image_keys.items():
            if obs_key in observation:
                img = observation[obs_key]  # H, W, C, uint8
                img_tensor = (
                    torch.from_numpy(img).to(self.device).to(target_dtype) / 255.0
                )
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                inputs[lerobot_key] = img_tensor

        # Process state if required
        if (
            hasattr(self.policy.config, "input_features")
            and "observation.state" in self.policy.config.input_features
        ):
            if "state" in observation:
                state = observation["state"]
                # Pad state to max_state_dim if needed
                if len(state) < self.max_state_dim:
                    padded = np.zeros(self.max_state_dim, dtype=state.dtype)
                    padded[:len(state)] = state
                    state = padded
                state_tensor = (
                    torch.from_numpy(state)
                    .to(self.device)
                    .to(target_dtype)
                    .unsqueeze(0)
                )
                inputs["observation.state"] = state_tensor

        # Tokenize task description
        if self.tokenizer:
            # XVLA expects clean task description
            cleaned_text = task_description.strip()

            tokens = self.tokenizer(
                [cleaned_text],
                return_tensors="pt",
                max_length=self.tokenizer_max_length,
                padding=self.pad_language_to,
                truncation=True,
            )
            inputs["observation.language.tokens"] = tokens["input_ids"].to(self.device)
            if "attention_mask" in tokens:
                inputs["observation.language.attention_mask"] = tokens["attention_mask"].to(
                    self.device, dtype=torch.bool
                )

        return inputs


class PI0FastAdapter(VLAAdapter):
    """
    Adapter for PI0Fast models.

    PI0Fast requires a different input format than PI0:
    - State is embedded in the text prompt as discretized values
    - Format: "Task: {task}, State: {state_str};\n"
    - State is normalized using mean/std and discretized into 256 bins
    """

    def __init__(self, policy, device: str):
        super().__init__(policy, device)
        # PI0Fast uses longer tokenizer max length
        self.tokenizer_max_length = getattr(policy.config, "tokenizer_max_length", 200)
        self.max_state_dim = getattr(policy.config, "max_state_dim", 32)

        # Try to get normalization stats from the policy
        self.state_mean = None
        self.state_std = None
        self._extract_normalization_stats()

    def _extract_normalization_stats(self):
        """Extract state normalization stats from the policy's preprocessor if available."""
        try:
            # Check if policy has preprocessors with stats
            if hasattr(self.policy, "preprocessor"):
                preprocessor = self.policy.preprocessor
                # Look for NormalizerProcessorStep in the pipeline
                if hasattr(preprocessor, "steps"):
                    for step in preprocessor.steps:
                        if hasattr(step, "stats") and step.stats is not None:
                            if "observation.state" in step.stats:
                                stats = step.stats["observation.state"]
                                if "mean" in stats:
                                    self.state_mean = stats["mean"].cpu().numpy()
                                if "std" in stats:
                                    self.state_std = stats["std"].cpu().numpy()
                                logger.info(
                                    f"Extracted state normalization stats from preprocessor"
                                )
                                return

            # Also check for dataset_stats attribute
            if hasattr(self.policy, "dataset_stats"):
                stats = self.policy.dataset_stats
                if stats and "observation.state" in stats:
                    state_stats = stats["observation.state"]
                    if "mean" in state_stats:
                        mean = state_stats["mean"]
                        self.state_mean = mean.cpu().numpy() if torch.is_tensor(mean) else mean
                    if "std" in state_stats:
                        std = state_stats["std"]
                        self.state_std = std.cpu().numpy() if torch.is_tensor(std) else std
                    logger.info("Extracted state normalization stats from dataset_stats")
                    return

            logger.warning(
                "PI0Fast: Could not extract state normalization stats. "
                "Using identity normalization (state values should already be normalized)."
            )
        except Exception as e:
            logger.warning(f"PI0Fast: Error extracting normalization stats: {e}")

    def _find_tokenizer(self):
        # Try standard find first
        tokenizer = super()._find_tokenizer()
        if tokenizer:
            # Ensure right padding for PI0Fast
            tokenizer.padding_side = "right"
            return tokenizer

        # Fallback to loading PaliGemma tokenizer (same as PI0)
        try:
            from transformers import AutoTokenizer

            model_id = "google/paligemma-3b-pt-224"
            logger.info(f"Loading fallback tokenizer for PI0Fast from {model_id}")
            # Use default PaliGemma tokenizer settings (add_bos_token=True, add_eos_token=False)
            # This matches the TokenizerProcessorStep in LeRobot's processor pipeline
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # PI0Fast uses right padding
            tokenizer.padding_side = "right"
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load fallback tokenizer for PI0Fast: {e}")
            return None

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using mean/std if available, otherwise clip to [-1, 1]."""
        if self.state_mean is not None and self.state_std is not None:
            # Use mean/std normalization
            mean = self.state_mean[:len(state)] if len(self.state_mean) > len(state) else self.state_mean
            std = self.state_std[:len(state)] if len(self.state_std) > len(state) else self.state_std

            # Pad mean/std if state is longer
            if len(state) > len(mean):
                mean = np.pad(mean, (0, len(state) - len(mean)), constant_values=0)
                std = np.pad(std, (0, len(state) - len(std)), constant_values=1)

            # Normalize: (x - mean) / std
            normalized = (state - mean) / (std + 1e-8)
            # Clip to [-1, 1] after normalization
            return np.clip(normalized, -1, 1)
        else:
            # Fallback: assume state is already somewhat normalized, just clip
            return np.clip(state, -1, 1)

    def _discretize_state(self, state: np.ndarray) -> str:
        """
        Discretize state into 256 bins and return as space-separated string.

        State should already be normalized to [-1, 1] range.
        """
        # Pad state to max_state_dim
        if state.shape[0] < self.max_state_dim:
            padded = np.zeros(self.max_state_dim)
            padded[:state.shape[0]] = state
            state = padded
        elif state.shape[0] > self.max_state_dim:
            state = state[:self.max_state_dim]

        # Discretize to 256 bins (same as PI0Fast processor)
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        discretized = np.digitize(state, bins=bins) - 1
        # Clamp to valid range
        discretized = np.clip(discretized, 0, 255)

        return " ".join(map(str, discretized))

    def prepare_inputs(
        self, observation: Dict[str, Any], task_description: str
    ) -> Dict[str, torch.Tensor]:
        """Convert RoboCerebra observation to PI0Fast input format with embedded state."""
        inputs = {}

        # Determine target dtype from policy parameters
        try:
            target_dtype = next(self.policy.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        # Process images
        for obs_key, lerobot_key in self.image_keys.items():
            if obs_key in observation:
                img = observation[obs_key]  # H, W, C, uint8
                img_tensor = (
                    torch.from_numpy(img).to(self.device).to(target_dtype) / 255.0
                )
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                inputs[lerobot_key] = img_tensor

        # Process state - PI0Fast embeds it in the prompt
        state = observation.get("state", None)
        if state is None:
            # Create zero state if not provided
            state = np.zeros(self.max_state_dim)

        # Normalize state using mean/std if available, otherwise clip to [-1, 1]
        state_normalized = self._normalize_state(state)

        # Store state tensor for the model (some PI0Fast models still use it)
        state_tensor = (
            torch.from_numpy(state)
            .to(self.device)
            .to(target_dtype)
            .unsqueeze(0)
        )
        inputs["observation.state"] = state_tensor

        # Create PI0Fast prompt with embedded state
        if self.tokenizer:
            # Clean task description (same as PI0Fast processor)
            cleaned_text = task_description.strip().replace("_", " ").replace("\n", " ")

            # Discretize state
            state_str = self._discretize_state(state_normalized)

            # Format prompt
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"

            tokens = self.tokenizer(
                [full_prompt],
                return_tensors="pt",
                max_length=self.tokenizer_max_length,
                padding="max_length",
                truncation=True,
            )
            inputs["observation.language.tokens"] = tokens["input_ids"].to(self.device)
            inputs["observation.language.attention_mask"] = tokens["attention_mask"].to(
                self.device, dtype=torch.bool
            )

        return inputs


# =============================================================================
# MAIN MODEL CLASS
# =============================================================================


class LerobotModel(RoboCerebraModel):
    def __init__(self):
        self.policy = None
        self.adapter = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, cfg: GenerateConfig) -> None:
        if not cfg.lerobot_checkpoint:
            raise ValueError(
                "lerobot_checkpoint must be specified in config for 'lerobot' model_family"
            )

        logger.info(f"Loading Lerobot policy from {cfg.lerobot_checkpoint}")

        # 1. Infer Policy Type
        type_name = self._infer_policy_type(cfg.lerobot_checkpoint)
        logger.info(f"Inferred policy type: {type_name}")

        if type_name == "openvla":
            logger.warning(
                "Detected 'openvla' policy type. LeRobot generally does not support OpenVLA natively. "
                "Ensure you are using a compatible LeRobot checkpoint or consider using model_family='openvla'."
            )

        # 2. Get Policy Class
        try:
            policy_cls = get_policy_class(type_name)
        except ValueError as e:
            # If policy type not found in registry (e.g. openvla, groot)
            logger.warning(
                f"Policy type '{type_name}' not in LeRobot registry. Attempting Generic/ACT fallback or re-raise."
            )
            if "act" in type_name.lower():
                policy_cls = get_policy_class("act")
            else:
                raise ValueError(
                    f"Unsupported LeRobot policy type: {type_name}. Error: {e}"
                )

        # 3. Load Policy
        try:
            # We pass compile_model=False as an override if supported
            try:
                self.policy = policy_cls.from_pretrained(
                    cfg.lerobot_checkpoint, compile_model=False
                )
            except TypeError:
                self.policy = policy_cls.from_pretrained(cfg.lerobot_checkpoint)
                if hasattr(self.policy.config, "compile_model"):
                    self.policy.config.compile_model = False
        except Exception as e:
            if "PaliGemma" in str(e) or "transformers" in str(e):
                logger.error(
                    "Error loading model. This is likely due to incompatible transformers version for PI0/PaliGemma. "
                    "Please upgrade transformers >= 4.41.0."
                )
            raise e

        self.policy.eval()
        self.policy.to(self.device)

        # Apply universal linear layer patch to handle hardcoded float32 in LeRobot
        self._patch_linear_layers(self.policy)

        # FIX: Ensure consistent dtype across the policy to avoid LayerNorm errors
        # (expected scalar type Float but found BFloat16)
        # We default to float32 unless the model is explicitly intended for bfloat16
        if self.device == "cuda":
            # Check if any parameter is bfloat16, if so, maybe keep it or force float32
            # For PI0, it's often better to stay in bfloat16 if it was loaded as such
            # but ensure EVERY parameter is in that dtype.
            has_bf16 = any(p.dtype == torch.bfloat16 for p in self.policy.parameters())
            if has_bf16:
                logger.info(
                    "Detected BFloat16 parameters. Ensuring policy is consistently BFloat16."
                )
                self.policy.to(torch.bfloat16)
            else:
                self.policy.to(torch.float32)
        else:
            self.policy.to(torch.float32)

        # 4. Initialize Adapter
        self._init_adapter(type_name)

    def _patch_linear_layers(self, model: torch.nn.Module):
        """
        Universal fix for linear layers to handle hardcoded float32 in LeRobot policies.
        This patches the forward method of all nn.Linear modules to cast input to weight dtype.
        """
        patch_count = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                old_fwd = m.forward

                def make_new_fwd(m_instance, old_f):
                    def new_fwd(x):
                        if torch.is_tensor(x) and x.dtype != m_instance.weight.dtype:
                            x = x.to(m_instance.weight.dtype)
                        return old_f(x)

                    return new_fwd

                m.forward = make_new_fwd(m, old_fwd)
                patch_count += 1
        logger.info(f"Applied dtype-consistency patch to {patch_count} linear layers")

    def _infer_policy_type(self, checkpoint_path: str) -> str:
        """Robustly infer policy type from config.json."""
        try:
            if Path(checkpoint_path).is_dir():
                config_file = Path(checkpoint_path) / CONFIG_NAME
            else:
                config_file = hf_hub_download(
                    repo_id=checkpoint_path, filename=CONFIG_NAME
                )

            with open(config_file, "r") as f:
                cfg_dict = json.load(f)

            type_name = cfg_dict.get("type") or cfg_dict.get("model_type")

            # Heuristics for missing type
            if not type_name and "architectures" in cfg_dict:
                archs = cfg_dict["architectures"]
                if isinstance(archs, list) and len(archs) > 0:
                    arch = archs[0]
                    if "Gr00t" in arch:
                        type_name = "groot"
                    elif "OpenVLA" in arch:
                        type_name = "openvla"
                    elif "PaliGemma" in arch:
                        type_name = "pi0"
                    else:
                        type_name = arch.lower()

            # Aliases
            if type_name == "Gr00tN1d6":
                type_name = "groot"

            if not type_name:
                raise ValueError("Could not determine policy type from config")

            return type_name

        except Exception as e:
            logger.warning(
                f"Failed to infer policy type via config: {e}. Defaulting to 'act'."
            )
            return "act"

    def _init_adapter(self, type_name: str):
        if type_name == "pi0_fast":
            self.adapter = PI0FastAdapter(self.policy, self.device)
        elif type_name in ["pi0", "pi05"]:
            self.adapter = PI0Adapter(self.policy, self.device)
        elif type_name == "smolvla":
            self.adapter = SmolVLAAdapter(self.policy, self.device)
        elif type_name == "xvla":
            self.adapter = XVLAAdapter(self.policy, self.device)
        elif type_name == "groot":
            self.adapter = VLAAdapter(self.policy, self.device)
        elif "openvla" in type_name:
            self.adapter = VLAAdapter(self.policy, self.device)
        elif "act" in type_name or "diffusion" in type_name:
            self.adapter = PolicyAdapter(self.policy, self.device)
        else:
            # Fallback: Check for language features
            if hasattr(self.policy.config, "input_features") and any(
                "language" in k for k in self.policy.config.input_features
            ):
                self.adapter = VLAAdapter(self.policy, self.device)
            else:
                self.adapter = PolicyAdapter(self.policy, self.device)

        logger.info(f"Initialized adapter: {self.adapter.__class__.__name__}")

    def get_image_size(self) -> Union[int, tuple]:
        return 224

    def predict_action(
        self, observation: Dict[str, Any], task_description: str
    ) -> np.ndarray:
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.adapter.prepare_inputs(observation, task_description)

        with torch.no_grad():
            action = self.policy.select_action(inputs)

        return self.adapter.post_process_action(action)

    def reset(self) -> None:
        if self.adapter:
            self.adapter.reset()
