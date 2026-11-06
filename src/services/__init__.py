"""
Business services for PharmaSupply-ML.

Services contain framework-independent logic. ``inventory`` implements alert
classification, replenishment arithmetic, FEFO loss estimation and the shared
feature vector builder. ``model_status`` reads the drift status file and runs
the transition-based e-mail watcher. This separation allows unit tests to
exercise business rules without FastAPI, PostgreSQL or SMTP.
"""
