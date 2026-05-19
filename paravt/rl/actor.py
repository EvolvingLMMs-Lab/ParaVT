"""Hierarchical PPO Actor for main + subagent trajectory training"""

from typing import Any

import torch
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.utils.data import concat_padded_tensors


class HierarchicalPPOActor(FSDPPPOActor):
    """Actor for hierarchical agent training with separate subagent trajectories

    This actor handles training data where:
    - Main trajectory data is in the main dict (input_ids, logprobs, etc.)
    - Subagent trajectories are stored in 'subagent_trajectories' as a list
    - When include_subagent_in_loss=True: main + subagent trajectories are concatenated for training
    - When include_subagent_in_loss=False (default): only main trajectories are trained
    - Advantages are computed only on main trajectories, then broadcasted to subagents
    """

    def __init__(self, config, include_subagent_in_loss: bool = False, **kwargs):
        super().__init__(config=config, **kwargs)
        self.include_subagent_in_loss = include_subagent_in_loss

    @torch.no_grad()
    def compute_logp(self, data: dict[str, Any]) -> torch.Tensor:
        """Compute logprobs for main (+ optionally subagent) trajectories"""
        if "subagent_trajectories" not in data or not data["subagent_trajectories"]:
            return self.actor.compute_logp(data)

        if not self.include_subagent_in_loss:
            # Skip subagent: only compute logp on main trajectory
            saved_sub = data.pop("subagent_trajectories")
            result = self.actor.compute_logp(data)
            data["subagent_trajectories"] = saved_sub
            return result

        # Original behavior: concat main + subagent trajectories
        # Pop keys that only exist in main data (not in subagent trajectories)
        # to avoid KeyError during concat_padded_tensors
        saved_keys = {}
        for key in ["prox_logp", "ref_logp"]:
            if key in data:
                saved_keys[key] = data.pop(key)

        all_trajectories = [data] + data["subagent_trajectories"]
        combined_data = concat_padded_tensors(all_trajectories)
        combined_data.pop("subagent_trajectories")
        result = self.actor.compute_logp(combined_data)

        # Restore popped keys to original data dict
        data.update(saved_keys)

        return result

    @torch.no_grad()
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        if "subagent_trajectories" not in data or not data["subagent_trajectories"]:
            return self.actor.compute_advantages(data)

        subagent_trajectories = data["subagent_trajectories"]
        main_batch_size = data["input_ids"].shape[0]

        # ref_logp may have combined batch size (main + subagent) from compute_logp.
        # Slice to main-only for parent compute_advantages which expects matching sizes.
        if "ref_logp" in data and data["ref_logp"].shape[0] > main_batch_size:
            data["ref_logp"] = data["ref_logp"][:main_batch_size].clone()

        adv_batch = self.actor.compute_advantages(data)

        if not self.include_subagent_in_loss:
            # Skip subagent: return main-only advantages, no concat needed
            adv_batch["subagent_trajectories"] = subagent_trajectories
            return adv_batch

        # Original behavior: broadcast advantages to subagents and concat
        episode_ids = data["episode_id"].detach().cpu().tolist()

        episode_id2values = {}
        for key in ["advantages", "returns", "kl_rewards", "tot_rewards"]:
            values = adv_batch[key]
            episode_id2values[key] = {}
            assert len(values) == len(episode_ids)
            for episode_id, value in zip(episode_ids, values, strict=False):
                episode_id2values[key][episode_id] = value

        for subagent_trajectory in subagent_trajectories:
            episode_id = subagent_trajectory["episode_id"].item()
            batch_size = subagent_trajectory["input_ids"].shape[0]
            max_seqlen = subagent_trajectory["input_ids"].shape[1]

            for key in ["advantages", "returns", "kl_rewards", "tot_rewards"]:
                main_value = episode_id2values[key][episode_id]
                if main_value.dim() == 0:
                    subagent_trajectory[key] = main_value.expand(batch_size, max_seqlen)
                elif main_value.dim() == 1:
                    subagent_trajectory[key] = main_value[0].expand(
                        batch_size, max_seqlen
                    )
                else:
                    subagent_trajectory[key] = main_value[0].expand(
                        batch_size, max_seqlen
                    )

        # Pop keys that only exist in main data before concat_padded_tensors
        proxy_logp = data.pop("prox_logp")
        # ref_logp may be present from an upstream compute_logp pass; pop so
        # concat_padded_tensors does not require all subagent trajectories to
        # carry it. The trainer recomputes ref_logp on the combined batch via
        # ref.compute_logp(rollout_batch), so we do not need to preserve it here.
        data.pop("ref_logp", None)

        all_trajectories = [data] + data["subagent_trajectories"]
        combined_data = concat_padded_tensors(all_trajectories)
        combined_data["prox_logp"] = proxy_logp
        return combined_data
