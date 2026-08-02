"""MK0-v1 contract tests.

The tests in this package are deliberately CPU-only.  They exercise exact
combinatorics, deterministic randomized properties and numerical oracles; the
contract's neural forward/backward acceptance is run by the separate GPU
runner and is never silently substituted here.
"""
