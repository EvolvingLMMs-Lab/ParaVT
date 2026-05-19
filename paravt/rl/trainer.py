"""ParaVT-flavored PPOTrainer.

Wraps AReaL's :class:`PPOTrainer` to plug in a hierarchical PPO actor that
knows how to backprop through tool-call sub-trajectories. Kept in its own
module so the orchestration script (:mod:`paravt.rl.train`) stays small and
the trainer subclass is easy to extend (custom checkpointing, eval hooks,
etc.) without recompiling the entry point.

This trainer subclasses areal.experimental.trainer.PPOTrainer. Upstream
AReaL has not committed to the experimental API surface, so a future bump
of the vendored AReaL pin (see patches/areal/README.md) may require
minor adjustments to this module.
"""

from __future__ import annotations

from areal.api.cli_args import PPOActorConfig
from areal.experimental.trainer import PPOTrainer

from paravt.rl.actor import HierarchicalPPOActor


class HierarchicalPPOTrainer(PPOTrainer):
    """:class:`PPOTrainer` that builds a :class:`HierarchicalPPOActor`."""

    def _create_actor(self, actor_config: PPOActorConfig) -> HierarchicalPPOActor:
        actor = HierarchicalPPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor
