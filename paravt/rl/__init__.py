"""ParaVT RL training subpackage.

Imports are kept lazy: pulling in :class:`HierarchicalAgentGRPOConfig`
or anything from :mod:`paravt.rl.train` requires the RL venv (AReaL,
SGLang, hydra). The :mod:`paravt.eval` venv must remain importable
without those deps, so this ``__init__`` deliberately re-exports
nothing.
"""
