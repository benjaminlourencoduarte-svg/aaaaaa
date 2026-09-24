import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from brain import TinyBrain
from pong_game import PongGame

ACTIONS = {"up": 5001, "stay": 5002, "down": 5003}

PROJECT_DIR = Path(__file__).resolve().parent


def start_neurons():
    processes = []

    neuron_script = PROJECT_DIR / "neuron_service.py"
    weights_dir = PROJECT_DIR / ".brain_weights"

    for action, port in ACTIONS.items():
        weight_file = weights_dir / f"{action}.json"

        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(neuron_script),
                    action,
                    str(port),
                    str(weight_file),
                ],
                cwd=str(PROJECT_DIR),
            )
        )

    return processes
