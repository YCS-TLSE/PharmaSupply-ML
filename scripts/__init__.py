"""
Operational helper scripts for PharmaSupply-ML.

Scripts in this package are invoked by the CLI or by ``run_pipeline.py``.
They handle artifact cleanup and environment-file conversion. They must stay
side-effect free at import time so they can be imported safely by tests or by
the orchestrator.
"""
