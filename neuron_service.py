"""Flask process used as one localhost neuron.

Each process exposes a JSON endpoint. The controller's HTTP POST is the signal
travelling across a local connection. We keep this service deliberately simple:
a neuron computes a weighted sum and learns from reward feedback.
"""

from flask import Flask, jsonify, request
import json
import os
import random
import traceback

FEATURES = ["ball_x", "ball_y", "ball_dx", "ball_dy", "paddle_y"]


def create_app(action, weight_file=None):
    app = Flask(__name__)
    weights = load_weights(weight_file)

    @app.post("/signal")
    def signal():
        try:
            observation = request.get_json(force=True) or {}
            # The bias and weighted observation are this neuron's activation.
            score = weights["bias"] + sum(
                weights[name] * float(observation.get(name, 0.0)) for name in FEATURES
            )
            return jsonify({"action": action, "score": score})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc), "action": action, "score": 0.0}), 400

    @app.post("/feedback")
    def feedback():
        try:
            body = request.get_json(force=True) or {}
            reward = float(body.get("reward", 0.0))
            observation = body.get("observation", {})
            learning_rate = float(body.get("learning_rate", 0.05))
            # Reward strengthens the recent connection pattern; punishment weakens it.
            direction = 1.0 if reward > 0 else -1.0
            for name in FEATURES:
                value = max(-1.0, min(1.0, float(observation.get(name, 0.0))))
                weights[name] += learning_rate * direction * abs(reward) * value
            weights["bias"] += learning_rate * direction * abs(reward) * 0.1
            save_weights(weight_file, weights)
            return jsonify({"ok": True, "action": action, "weights": weights})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 400

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "action": action})

    return app


def load_weights(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            traceback.print_exc()
    return {"bias": random.uniform(-0.1, 0.1), **{name: random.uniform(-1, 1) for name in FEATURES}}


def save_weights(path, weights):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(weights, file, indent=2)


def run_neuron(action, port, weight_file):
    # Binding to 127.0.0.1 ensures this experiment is not exposed to a network.
    create_app(action, weight_file).run(host="127.0.0.1", port=port, threaded=False, use_reloader=False)
