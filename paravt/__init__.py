"""ParaVT: Parallel Agentic Video Tools.

A hierarchical agent framework for long-video understanding, post-trained
with PARA-GRPO (Parseability-Anchored and Ratio-gAted GRPO) — two
components: Exploration Anchoring rewards parseable format only at the
structural tokens most prone to collapse, and nFrames Gating randomizes
the per-prompt overview frame budget so calling the tool earns
measurable credit when overview frames alone are insufficient.

The training submodules (:mod:`paravt.rl.workflow`, :mod:`paravt.rl.train`,
:mod:`paravt.rl.trainer`) pull in the AReaL stack at import time, which is
heavy and only useful in an RL environment. The evaluation submodules
(:mod:`paravt.eval.utils`, :mod:`paravt.eval.driver`) only need a vLLM
environment. To keep ``import paravt`` cheap and broadly compatible,
this module re-exports nothing eagerly — pull what you need by full
submodule path::

    from paravt.rl.workflow import HierarchicalAgentWorkflow, HierarchicalAgentConfig
    from paravt.rl.subagents.manager import SubagentManager
    from paravt.rl.subagents.base import SubagentToolBase
    from paravt.eval.driver import eval_one

This avoids the eval venv (vLLM-only) needing hydra/AReaL and the RL
venv (sglang+AReaL) needing vLLM.
"""

__version__ = "0.1.0"
