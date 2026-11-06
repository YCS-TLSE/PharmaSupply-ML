"""
===============================================================================
PharmaSupply-ML: environment.yml → requirements.txt Converter
===============================================================================

PURPOSE
-------
Keep the pip-friendly ``requirements.txt`` aligned with the conda environment
definition by extracting conda string dependencies and the ``pip:`` subsection.

RESPONSIBILITIES
----------------
* Load ``environment.yml`` from the repository working directory.
* Skip the Python interpreter pin itself.
* Copy conda package specs and pip package specs into ``requirements.txt``.

INPUTS / OUTPUTS
----------------
Inputs: ``environment.yml`` in the current working directory.
Outputs: ``requirements.txt`` overwritten with one dependency per line.

ARCHITECTURE INTEGRATION
------------------------
Maintenance utility used when the conda environment changes. Docker images use
the dedicated ``requirements.api.txt`` / ``requirements.dashboard.txt`` files,
so this script is primarily a developer convenience.

OPERATIONAL NOTE
----------------
The script deliberately executes at import time to preserve its historical CLI
behaviour. Run it from the repository root.
"""

import yaml

# Load the declarative conda environment.
with open("environment.yml", "r") as f:
    env = yaml.safe_load(f)

requirements = []

# Conda dependencies can be plain strings or a mapping containing a pip list.
for dep in env.get("dependencies", []):
    if isinstance(dep, str):
        # Exclude the Python interpreter itself; pip/venv manages it.
        if not dep.startswith("python"):
            requirements.append(dep)
    elif isinstance(dep, dict) and "pip" in dep:
        # Preserve pip-only packages (FastAPI, MLflow, tooling, etc.).
        requirements.extend(dep["pip"])

# Write the flattened dependency list consumed by pip-based setups.
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements) + "\n")

print(
    "requirements.txt généré avec succès depuis environment.yml !"
)
