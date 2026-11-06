"""
Persistence layer for PharmaSupply-ML.

``database`` owns the process-wide SQLAlchemy engine, the connection pool and
the cursor context manager. ``inventory`` contains read-only SQL used by the
serving layer. Keeping SQL outside the routers makes the API testable with
stub cursors and prevents connection-management logic from leaking into HTTP
handlers.
"""
