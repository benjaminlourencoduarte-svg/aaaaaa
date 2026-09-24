"""Localhost neuron Pong: one-file version.

This file contains:
- a tiny Pong game
- a lightweight neural controller using HTTP calls to local Flask services
- a Flask app that behaves like a neuron receiving observations and giving output
- a simple reward-based learning loop

The project remains intentionally small, readable, and local-only.
"""

import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import requests
from flask import Flask, jsonify, request

try:
    import pygame
except Exception:
    pygame = None


FEATURES = ["ball_x", "ball_y", "ball_dx", "ball_dy", "paddle_y"]
ACTIONS = {"up": 5001, "stay": 5002, "down": 5003}
PROJECT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pong environment
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    ball_x: float
    ball_y: float
    ball_dx: float
    ball_dy: float
    paddle_y: float

    def as_dict(self):
        return {
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_dx": self.ball_dx,
            "ball_dy": self.ball_dy,
            "paddle_y": self.paddle_y,
        }


class PongGame:
    WIDTH, HEIGHT = 640, 400
    PADDLE_W, PADDLE_H = 14, 80
    BALL_SIZE = 12

    def __init__(self):
        self.paddle_x = 35
        self.paddle_speed = 6
        self.reset()

    def reset(self):
        self.paddle_y = self.HEIGHT / 2 - self.PADDLE_H / 2
        self.ball_x = self.WIDTH / 2
        self.ball_y = random.uniform(60, self.HEIGHT - 60)
        self.ball_dx = random.choice([-4.0, 4.0])
        self.ball_dy = random.choice([-3.0, 3.0])
        return self.observation()

    def observation(self):
        return Observation(
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.ball_dx / 4.0,
            self.ball_dy / 3.0,
            (self.paddle_y + self.PADDLE_H / 2) / self.HEIGHT,
        )

    def step(self, action):
        """Move paddle and ball; return (observation, reward, done, hit)."""
        if action == "up":
            self.paddle_y -= self.paddle_speed
        elif action == "down":
            self.paddle_y += self.paddle_speed
        self.paddle_y = max(0, min(self.HEIGHT - self.PADDLE_H, self.paddle_y))

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        if self.ball_y <= 0 or self.ball_y + self.BALL_SIZE >= self.HEIGHT:
            self.ball_dy *= -1

        hit = False
        reward = 0.0

        if self.ball_dx < 0 and self.ball_x <= self.paddle_x + self.PADDLE_W:
            if self.paddle_y <= self.ball_y <= self.paddle_y + self.PADDLE_H:
                self.ball_x = self.paddle_x + self.PADDLE_W
                self.ball_dx = abs(self.ball_dx) * 1.03
                hit = True
                reward = 1.0

        missed = self.ball_x < -self.BALL_SIZE
        if missed:
            reward = -2.0
        elif self.ball_x > self.WIDTH:
            self.ball_x = self.WIDTH - self.BALL_SIZE
            self.ball_dx *= -1

        return self.observation(), reward, missed, hit


# ---------------------------------------------------------------------------
# Neuron logic: Flask app representing one local service
# ---------------------------------------------------------------------------

def load_weights(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            traceback.print_exc()

    return {
        "bias": random.uniform(-0.1, 0.1),
        **{name: random.uniform(-1.0, 1.0) for name in FEATURES},
    }


def save_weights(path, weights):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(weights, handle, indent=2)


def create_neuron_app(action, weight_file=None):
    """Create a Flask app that acts like a neuron.

    Each localhost service is a tiny neuron-like unit. It exposes small HTTP
    endpoints. The controller sends observations via POST /signal and receives
    a scalar score. Reward feedback via POST /feedback updates connection weights.
    """
    app = Flask(__name__)
    weights = load_weights(weight_file)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "action": action})

    @app.post("/signal")
    def signal():
        try:
            observation = request.get_json(force=True) or {}
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
            payload = request.get_json(force=True) or {}
            reward = float(payload.get("reward", 0.0))
            observation = payload.get("observation", {})
            learning_rate = float(payload.get("learning_rate", 0.05))

            # This is the educational version of adjusting connection strengths.
            # A successful hit increases weights that led to the current action;
            # a miss decreases them.
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

    return app


def run_neuron_service(action, port, weight_file):
    """Run one local Flask neuron service at localhost only."""
    create_neuron_app(action, weight_file).run(
        host="127.0.0.1",
        port=port,
        threaded=False,
        use_reloader=False,
    )


# ---------------------------------------------------------------------------
# Controller: HTTP requests represent signals across local neuron connections
# ---------------------------------------------------------------------------

class TinyBrain:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.timeout = 0.20

    def choose_action(self, observation):
        scores = {}
        for action, url in self.endpoints.items():
            try:
                response = requests.post(url + "/signal", json=observation, timeout=self.timeout)
                response.raise_for_status()
                scores[action] = float(response.json().get("score", 0.0))
            except requests.RequestException:
                traceback.print_exc()
                scores[action] = 0.0
        return max(scores, key=scores.get), scores

    def learn(self, action, observation, reward):
        try:
            response = requests.post(
                self.endpoints[action] + "/feedback",
                json={
                    "observation": observation,
                    "reward": reward,
                    "learning_rate": 0.05,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Launcher: game loop and neuron child process
# ---------------------------------------------------------------------------

def start_neurons():
    processes = []
    for action, port in ACTIONS.items():
        weight_file = PROJECT_DIR / ".brain_weights" / f"{action}.json"
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_DIR / "main.py"),
                    "--neuron",
                    action,
                    str(port),
                    str(weight_file),
                ],
                cwd=str(PROJECT_DIR),
            )
        )
    return processes


def draw_game(screen, game, font, episode, hits, total_reward):
    if screen is None or font is None:
        return
    screen.fill((15, 20, 35))
    pygame.draw.rect(screen, (230, 230, 240), (game.paddle_x, game.paddle_y, game.PADDLE_W, game.PADDLE_H))
    pygame.draw.circle(screen, (80, 220, 150), (int(game.ball_x), int(game.ball_y)), game.BALL_SIZE // 2)
    text = f"Episode {episode}  Score {hits}  Reward {total_reward:.1f}  Local neurons learning"
    screen.blit(font.render(text, True, (240, 240, 240)), (10, 10))
    pygame.display.flip()


def run_game_loop():
    processes = start_neurons()
    screen = None
    font = None
    clock = None
    game = PongGame()
    brain = TinyBrain({action: f"http://127.0.0.1:{port}" for action, port in ACTIONS.items()})
    episode = 1
    total_reward = 0.0
    hits = 0
    frames = 0

    try:
        time.sleep(1.0)

        if os.environ.get("HEADLESS") != "1" and pygame is not None:
            pygame.init()
            screen = pygame.display.set_mode((game.WIDTH, game.HEIGHT))
            pygame.display.set_caption("Localhost Neuron Pong")
            clock = pygame.time.Clock()
            font = pygame.font.Font(None, 24)

        running = True
        while running:
            if screen and pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            observation = game.observation().as_dict()
            action, scores = brain.choose_action(observation)
            next_observation, reward, done, hit = game.step(action)
            brain.learn(action, observation, reward)
            total_reward += reward
            hits += int(hit)
            frames += 1

            if screen and pygame is not None:
                draw_game(screen, game, font, episode, hits, total_reward)
                clock.tick(60)
            elif frames % 60 == 0:
                print(f"episode={episode} score={hits} reward={total_reward:.1f} last={action} scores={scores}")

            if done:
                episode += 1
                total_reward = 0.0
                hits = 0
                game.reset()

    except KeyboardInterrupt:
        print("Stopping experiment...")
    except Exception:
        traceback.print_exc()
    finally:
        if screen is not None and pygame is not None:
            pygame.quit()
        for process in processes:
            try:
                process.terminate()
            except Exception:
                pass
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--neuron":
        if len(sys.argv) < 5:
            raise SystemExit("Usage: main.py --neuron <action> <port> <weight_file>")
        action = sys.argv[2]
        port = int(sys.argv[3])
        weight_file = sys.argv[4]
        run_neuron_service(action, port, weight_file)
        return

    run_game_loop()


if __name__ == "__main__":
    main()
