"""Packaging and dependency installer for the one-file project."""

from setuptools import setup

setup(
    name="localhost-neuron-pong",
    version="1.2.0",
    description="Educational localhost neuron Pong and AI SDK example",
    py_modules=["main"],
    python_requires=">=3.9",
    install_requires=[
        "Flask>=3.0,<4",
        "requests>=2.31,<3",
        "ai",
    ],
)
