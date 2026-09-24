"""Packaging and dependency installer for Localhost Neuron Pong.

Running ``python setup.py install`` is supported for this small educational
project, but modern Python projects should normally use:

    python -m pip install .

The install_requires list tells pip to download Flask, Pygame, and Requests.
"""

from pathlib import Path
from setuptools import setup


ROOT = Path(__file__).parent

setup(
    name="localhost-neuron-pong",
    version="1.0.0",
    description="A beginner-friendly Pong experiment controlled by localhost Flask neurons",
    py_modules=["main", "brain", "pong_game", "neuron_service"],
    python_requires=">=3.9",
    install_requires=[
        "Flask>=3.0,<4",
        "pygame>=2.5,<3",
        "requests>=2.31,<3",
    ],
)
