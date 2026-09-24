# Localhost Neuron Pong

A small educational experiment: three Flask processes act as localhost-only output neurons and control a simple Pong paddle. The controller sends observations over HTTP, chooses the strongest action, and rewards or penalizes the action neurons after each rally.

This is intentionally small and not biologically realistic. HTTP requests are the signals, and each neuron's JSON weight vector represents the strength of its incoming connections.

## Install

Python 3.9+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
