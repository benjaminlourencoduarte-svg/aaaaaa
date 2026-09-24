# Localhost Neuron Pong

A small educational experiment: three Flask processes act as localhost-only output neurons and control a simple Pong paddle. The controller sends observations over HTTP, chooses the strongest action, and rewards or penalizes the action neurons after each rally.

This is intentionally small and not biologically realistic. HTTP requests are the signals, and each neuron's JSON weight vector represents the strength of its incoming connections.

## Download this README with Git

Git downloads repository files by cloning the repository. To download only `README.md` without checking out the other files, use Git's sparse checkout:

```bash
git clone --filter=blob:none --no-checkout https://github.com/benjaminlourencoduarte-svg/aaaaaa.git aaaaaa
cd aaaaaa
git sparse-checkout init --no-cone
git sparse-checkout set README.md
git checkout
```

The checked-out README will be available at `aaaaaa/README.md`.

## Install

Python 3.9+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

A Pygame window opens and the controller starts three Flask neuron services on `127.0.0.1` ports 5001-5003. Close the window to stop the services. The terminal and game window show the score, episode, reward, and learning progress.

To run without a graphical window, set `HEADLESS=1`:

```bash
HEADLESS=1 python main.py
```

## How it works

- `pong_game.py` contains the tiny Pong environment and reward events.
- `neuron_service.py` is one reusable Flask service. Each process is an action neuron (`up`, `stay`, or `down`).
- `brain.py` sends observations to all neurons with `requests`. A request is a signal traveling along a local connection.
- After a hit or miss, `/feedback` adjusts the selected neuron's weights. Hits reinforce the recent decision; misses weaken it.
- `main.py` starts the services, runs the learning loop, and displays useful progress.

All services bind explicitly to `127.0.0.1`; no external network access is needed. HTTP calls have short timeouts and failures are reported with a traceback while the game safely falls back to staying still.

## Educational safety note

This is a local toy controller. It does not collect data, open a public server, or attempt to model a real brain. Delete `.brain_weights/*.json` to reset learned weights.
