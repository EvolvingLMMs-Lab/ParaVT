# Minimal setup.py for the vendored AReaL package.
#
# Upstream inclusionAI/AReaL ships build metadata (pyproject.toml + setup.py)
# at the *repo root* and the `areal/` Python package under it. ParaVT's
# release vendors only the `areal/` package directory directly under
# paravt/rl/areal/, so this directory ends up being the package root rather
# than the repo root. uv / pip cannot install a package directory with no
# build system, so we provide this thin shim:
#
#   - Tell setuptools the directory IS the `areal` namespace (package_dir
#     maps "areal" -> ".").
#   - Recursively register every subdirectory with __init__.py as
#     `areal.<subpkg>` so `import areal.api.cli_args` etc. resolve.
#
# When upstream is upgraded, regenerate the package list by re-running
# `setuptools.find_packages(where=".")` from this directory; nothing else
# needs to change.
from pathlib import Path

from setuptools import find_namespace_packages, setup

HERE = Path(__file__).parent.resolve()

# Discover every nested directory under this dir as a (namespace or
# regular) package and namespace them under `areal.`. AReaL upstream uses
# PEP 420 namespace packages for `api/`, `launcher/`, `models/`,
# `workflow/`, `experimental/`, `thirdparty/`, `tools/` (no __init__.py),
# while `controller/`, `core/`, `dataset/`, `engine/`, `platforms/`,
# `reward/`, `scheduler/`, `utils/` are regular packages — so a plain
# `find_packages` would miss the namespace ones. We use
# `find_namespace_packages` and exclude the test tree from the install.
# The bare top-level `areal` package is added explicitly because
# find_namespace_packages can't see the dir it's being run from.
sub_packages = find_namespace_packages(where=str(HERE), exclude=["tests*"])

setup(
    name="areal",
    version="0.1.0",
    description="Vendored copy of inclusionAI/AReaL — see upstream for source.",
    package_dir={"areal": "."},
    packages=["areal"] + [f"areal.{p}" for p in sub_packages],
    include_package_data=True,
    python_requires=">=3.10",
)
