"""ParaVT RL training entry point.

This is the script invoked by ``areal.launcher.local`` (see
``scripts/run_rl.sh``). It does only orchestration:

1. Load and validate the YAML recipe.
2. Build the tokenizer, processor, and dataset.
3. Hand control to :class:`paravt.rl.trainer.HierarchicalPPOTrainer`.

All reward logic lives in :mod:`paravt.rl.rewards`; the workflow lives in
:mod:`paravt.rl.workflow`; the dataset adapter in :mod:`paravt.rl.data.dataset`.
This file is intentionally small so reviewers can take in the full
training control flow at a glance.
"""

from __future__ import annotations

# AReaL's distributed FSDP path is currently incompatible with cuDNN's
# Conv3d kernels under PyTorch 2.9.1; disable cuDNN before importing torch
# is loaded into worker processes. The SGLang server worker has an
# independent cuDNN setting controlled via ``requirements/rl.lock`` (we pin
# nvidia-cudnn-cu12==9.16.0.29 there).
import torch  # isort: skip
torch.backends.cudnn.enabled = False  # noqa: E402

import os  # noqa: E402
import sys  # noqa: E402

from areal.api.cli_args import load_expr_config  # noqa: E402
from areal.utils import logging  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

from paravt.rl.config import paravt_block_to_env  # noqa: E402
from paravt.rl.data.dataset import load_paravt_rl_dataset  # noqa: E402
from paravt.rl.rewards import compose_reward  # noqa: E402
from paravt.rl.rewards.llm_judge import get_tracker  # noqa: E402
from paravt.rl.trainer import HierarchicalPPOTrainer  # noqa: E402
from paravt.rl.workflow import HierarchicalAgentGRPOConfig  # noqa: E402

logger = logging.getLogger("ParaVT.Train")


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, HierarchicalAgentGRPOConfig)

    # scripts/run_rl.sh has already exported the YAML's paravt: block to
    # env vars before this script ran. Hydra CLI overrides on the
    # command line update config.paravt but cannot reach those env
    # exports. Re-export here from the merged config so reward modules
    # and the workflow helpers see the correct values.
    paravt_env = paravt_block_to_env(config.paravt)
    os.environ.update(paravt_env)
    logger.info(f"PARA-GRPO env (post-Hydra merge): {paravt_env}")

    reward_mode = os.environ.get("REWARD_MODE", "f1")
    video_root = os.environ.get("PARAVT_VIDEO_ROOT", "./data/videos")

    logger.info("Starting ParaVT hierarchical-agent RL training")
    logger.info(f"Reward mode: {reward_mode}")
    logger.info(f"Base model: {config.actor.path}")

    processor = AutoProcessor.from_pretrained(config.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    train_dataset = load_paravt_rl_dataset(
        config.train_dataset.path, video_root=video_root
    )
    valid_dataset = None  # Validation is currently driven by the eval recipe.

    workflow_kwargs = dict(
        reward_fn=compose_reward,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        config=config.hierarchical_agent,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    # Slightly lower temperature for evaluation rollouts.
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    try:
        with HierarchicalPPOTrainer(config, train_dataset, valid_dataset) as trainer:
            trainer.train(
                workflow="paravt.rl.workflow.HierarchicalAgentWorkflow",
                workflow_kwargs=workflow_kwargs,
                eval_workflow="paravt.rl.workflow.HierarchicalAgentWorkflow",
                eval_workflow_kwargs=eval_workflow_kwargs,
            )
    finally:
        if reward_mode == "llm":
            logger.info("Training finished — final LLM-judge usage:")
            get_tracker().log_final()


if __name__ == "__main__":
    main(sys.argv[1:])
