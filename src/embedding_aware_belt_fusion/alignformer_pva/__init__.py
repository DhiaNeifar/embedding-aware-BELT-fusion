"""Propose-Verify-Abstain association backend (box-geometry only, no embeddings).

A second, independent alternative to the PQ-STF embedding + pair-head
backend in ``embeddings/all_agent_association.py``. Built to run side by side
with it through the same evaluation harness, so the two can be compared under
identical noise injection, detector, and AP computation.
"""
