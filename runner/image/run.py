"""Thin launcher for the image-worker server.

Equivalent to ``python -m runner.image.server``; kept as a convenient parallel to
the other runners. The module must be invoked as a package (``python -m
runner.image.run``) so the relative ``runner.image.server`` import resolves.
"""
from runner.image.server import main

if __name__ == "__main__":
    main()
