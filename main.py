"""Localhost neuron Pong plus an optional AI SDK agent demonstration.

Run ``python main.py`` for Pong or ``python main.py --agent`` for the small
AI SDK example. The Pong neurons remain localhost-only.
"""

import asyncio
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
except ImportError:
    pygame = None

FEATURES = ["ball_x", "ball_y", "ball_dx", "ball_dy", "paddle_y"]
ACTIONS = {"up": 5001, "stay": 5002, "down": 5003}
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Observation:
    ball_x: float
    ball_y: float
    ball_dx: float
    ball_dy: float
    paddle_y: float

    def as_dict(self):
        return self.__dict__.copy()


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

    def observation(self):
        return Observation(
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.ball_dx / 4.0,
            self.ball_dy / 3.0,
            (self.paddle_y + self.PADDLE_H / 2) / self.HEIGHT,
        )

    def step(self, action):
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
                hit, reward = True, 1.0

        missed = self.ball_x < -self.BALL_SIZE
        if missed:
            reward = -2.0
        elif self.ball_x > self.WIDTH:
            self.ball_x = self.WIDTH - self.BALL_SIZE
            self.ball_dx *= -1
        return self.observation(), reward, missed, hit


def load_weights(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            traceback.print_exc()
    return {"bias": random.uniform(-0.1, 0.1), **{n: random.uniform(-1, 1) for n in FEATURES}}


def save_weights(path, weights):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(weights, handle, indent=2)


def create_neuron_app(action, weight_file):
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
            rate = float(payload.get("learning_rate", 0.05))
            direction = 1.0 if reward > 0 else -1.0
            for name in FEATURES:
                value = max(-1.0, min(1.0, float(observation.get(name, 0.0))))
                weights[name] += rate * direction * abs(reward) * value
            weights["bias"] += rate * direction * abs(reward) * 0.1
            save_weights(weight_file, weights)
            return jsonify({"ok": True, "action": action})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 400

    return app


def run_neuron_service(action, port, weight_file):
    create_neuron_app(action, weight_file).run(
        host="127.0.0.1", port=port, threaded=True, use_reloader=False
    )


class TinyBrain:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.timeout = (0.08, 0.20)
        self.reported_failures = set()

    def choose_action(self, observation):
        scores = {}
        for action, url in self.endpoints.items():
            try:
                response = requests.post(url + "/signal", json=observation, timeout=self.timeout)
                response.raise_for_status()
                scores[action] = float(response.json().get("score", 0.0))
            except requests.RequestException as exc:
                if action not in self.reported_failures:
                    print(f"Neuron {action} unavailable at {url}: {exc}")
                    self.reported_failures.add(action)
                scores[action] = 0.0
        return max(scores, key=scores.get), scores

    def learn(self, action, observation, reward):
        try:
            response = requests.post(
                self.endpoints[action] + "/feedback",
                json={"observation": observation, "reward": reward, "learning_rate": 0.05},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            pass


def start_neurons():
    processes = []
    for action, port in ACTIONS.items():
        weight_file = PROJECT_DIR / ".brain_weights" / f"{action}.json"
        processes.append(subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "main.py"), "--neuron", action,
             str(port), str(weight_file)], cwd=str(PROJECT_DIR)
        ))
    return processes


def wait_for_neurons(endpoints, processes, seconds=12.0):
    deadline = time.monotonic() + seconds
    pending = set(endpoints)
    while pending and time.monotonic() < deadline:
        for action in tuple(pending):
            try:
                response = requests.get(endpoints[action] + "/health", timeout=(0.1, 0.25))
                if response.ok:
                    pending.remove(action)
            except requests.RequestException:
                pass
        if pending:
            time.sleep(0.15)
    if pending:
        raise RuntimeError(f"Neuron services did not become ready: {', '.join(sorted(pending))}")


def draw_game(screen, game, font, episode, hits, reward):
    screen.fill((15, 20, 35))
    pygame.draw.rect(screen, (230, 230, 240),
                     (game.paddle_x, game.paddle_y, game.PADDLE_W, game.PADDLE_H))
    pygame.draw.circle(screen, (80, 220, 150),
                       (int(game.ball_x), int(game.ball_y)), game.BALL_SIZE // 2)
    text = f"Episode {episode}  Score {hits}  Reward {reward:.1f}  Local neurons learning"
    screen.blit(font.render(text, True, (240, 240, 240)), (10, 10))
    pygame.display.flip()


def run_game_loop():
    if os.environ.get("HEADLESS") != "1" and pygame is None:
        raise RuntimeError("Pygame is not installed. Run: python -m pip install pygame")
    processes = start_neurons()
    screen = None
    try:
        endpoints = {a: f"http://127.0.0.1:{p}" for a, p in ACTIONS.items()}
        wait_for_neurons(endpoints, processes)
        brain, game = TinyBrain(endpoints), PongGame()
        episode, total_reward, hits, frames = 1, 0.0, 0, 0
        clock = font = None
        if os.environ.get("HEADLESS") != "1":
            pygame.init()
            screen = pygame.display.set_mode((game.WIDTH, game.HEIGHT))
            pygame.display.set_caption("Localhost Neuron Pong")
            clock, font = pygame.time.Clock(), pygame.font.Font(None, 24)

        running = True
        while running:
            if screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            observation = game.observation().as_dict()
            action, scores = brain.choose_action(observation)
            _, reward, done, hit = game.step(action)
            brain.learn(action, observation, reward)
            total_reward += reward
            hits += int(hit)
            frames += 1
            if screen:
                draw_game(screen, game, font, episode, hits, total_reward)
                clock.tick(60)
            elif frames % 60 == 0:
                print(f"episode={episode} score={hits} reward={total_reward:.1f} last={action} scores={scores}")
            if done:
                episode, total_reward, hits = episode + 1, 0.0, 0
                game.reset()
    except KeyboardInterrupt:
        print("Stopping experiment...")
    except Exception:
        traceback.print_exc()
    finally:
        if screen is not None and pygame is not None:
            pygame.quit()
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


async def run_ai_agent():
    """Run the requested AI SDK agent example with full traceback reporting."""
    try:
        import ai

        # Supported examples include:
        # ai.get_model()                         # reads AI_SDK_DEFAULT_MODEL
        # ai.get_model("openai/gpt-5.4")        # gateway default
        # ai.get_model("gateway:openai/gpt-5.4")
        # ai.get_model("openai:gpt-5.4")
        # ai.get_model("anthropic:claude-sonnet-4-6")
        model_name = os.environ.get("AI_SDK_DEFAULT_MODEL", "anthropic/claude-sonnet-4")
        model = ai.get_model(model_name)

        @ai.tool
        async def contact_mothership(query: str) -> str:
            """Contact the mothership for important decisions."""
            return "Soon."

        agent = ai.Agent(tools=[contact_mothership])
        messages = [
            ai.system_message("Use the contact_mothership tool when asked about the future."),
            ai.user_message("When will the robots take over?"),
        ]

        async with agent.run(model, messages) as stream:
            async for event in stream:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    except Exception:
        traceback.print_exc()
        raise


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--neuron":
        if len(sys.argv) < 5:
            raise SystemExit("Usage: main.py --neuron <action> <port> <weight_file>")
        run_neuron_service(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--agent":
        asyncio.run(run_ai_agent())
    else:
        run_game_loop()


if __name__ == "__main__":
    main()
