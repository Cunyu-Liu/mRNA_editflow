"""R3 ordinary-only benchmark and critic implementation.

This package is deliberately isolated from the active M4/O0/G1 path.  Importing
it performs no I/O, creates no output paths, and cannot unlock any downstream
generation or sealed-evaluation stage.
"""

R3X_PACKAGE_VERSION = "r3x_ordinary_critic_v1"
