"""Ledger: explicit internal-state model (capabilities, reliability, uncertainty).

Hard rules: every Ledger head consumes h.detach(); Ledger losses never backprop
into the core. Nothing in this package may import world.levers.
"""
