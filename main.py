"""Pong controller and two-AI Pong demonstration.

Run ``python main.py`` for the localhost-neuron Pong controller.
Run ``python main.py --agent`` for the two-AI horizontal Pong match.

This version uses tkinter, included with standard Windows Python installations.
It does not import or require pygame.
"""

import asyncio
import json
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


# ---------------------------------------------------------------------------
# Localhost-neuron Pong mode, rendered with tkinter
# ---------------------------------------------------------------------------

def start_neurons():
    processes = []
    for action, port in ACTIONS.items():
        weight_file = PROJECT_DIR / ".brain_weights" / f"{action}.json"
        processes.append(subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "neuron_service.py"), action,
             str(port), str(weight_file)], cwd=str(PROJECT_DIR)
        ))
    return processes


def run():
    """Run the original local Flask-neuron Pong experiment without pygame."""
    import tkinter as tk

    processes = start_neurons()
    root = None
    try:
        time.sleep(1.0)
        endpoints = {a: f"http://127.0.0.1:{p}" for a, p in ACTIONS.items()}
        brain = TinyBrain(endpoints)
        game = PongGame()

        if os.environ.get("HEADLESS") == "1":
            while True:
                observation = game.observation().as_dict()
                action, _ = brain.choose_action(observation)
                _, reward, done, _ = game.step(action)
                brain.learn(action, observation, reward)
                if done:
                    game.reset()
            return

        root = tk.Tk()
        root.title("Localhost Neuron Pong")
        canvas = tk.Canvas(root, width=game.WIDTH, height=game.HEIGHT, bg="#0f1423")
        canvas.pack()
        status = tk.StringVar()
        tk.Label(root, textvariable=status).pack()

        state = {"episode": 1, "score": 0, "reward": 0.0, "running": True}

        def close():
            state["running"] = False
            root.destroy()

        def tick():
            if not state["running"]:
                return
            observation = game.observation().as_dict()
            action, scores = brain.choose_action(observation)
            _, reward, done, hit = game.step(action)
            brain.learn(action, observation, reward)
            state["reward"] += reward
            state["score"] += int(hit)

            canvas.delete("all")
            canvas.create_rectangle(
                game.paddle_x, game.paddle_y,
                game.paddle_x + game.PADDLE_W,
                game.paddle_y + game.PADDLE_H,
                fill="#eeeeee",
            )
            canvas.create_oval(
                game.ball_x - game.BALL_SIZE / 2,
                game.ball_y - game.BALL_SIZE / 2,
                game.ball_x + game.BALL_SIZE / 2,
                game.ball_y + game.BALL_SIZE / 2,
                fill="#50dc96",
            )
            status.set(
                f"Episode {state['episode']} | Score {state['score']} | "
                f"Reward {state['reward']:.1f} | Action: {action}"
            )

            if done:
                state["episode"] += 1
                state["score"] = 0
                state["reward"] = 0.0
                game.reset()
            root.after(16, tick)

        root.protocol("WM_DELETE_WINDOW", close)
        tick()
        root.mainloop()
    except KeyboardInterrupt:
        print("Stopping experiment...")
    except Exception:
        traceback.print_exc()
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


# ---------------------------------------------------------------------------
# Two-AI horizontal Pong mode
# ---------------------------------------------------------------------------

LEFT = "left - CRL"
RIGHT = "right - LRC"
STAY = "stay"
AI_PROMPT = (
    "You control one horizontal Pong paddle. Reply with exactly one action: "
    "left - CRL, right - LRC, or stay. Do not add punctuation or explanation. "
    "Use the JSON observation to move toward the ball."
)


class HorizontalPong:
    WIDTH, HEIGHT = 720, 440
    PADDLE_W, PADDLE_H = 110, 14
    BALL = 12

    def __init__(self):
        self.reset()

    def reset(self):
        self.ball_x, self.ball_y = self.WIDTH / 2, self.HEIGHT / 2
        self.ball_dx, self.ball_dy = 5.0, 4.0
        self.top_x = self.bottom_x = self.WIDTH / 2 - self.PADDLE_W / 2
        self.top_score = self.bottom_score = 0

    def observation(self, side, opponent_model, opponent_prompt):
        paddle_x = self.top_x if side == "top" else self.bottom_x
        opponent_x = self.bottom_x if side == "top" else self.top_x
        return {
            "side": side,
            "ball": {"x": round(self.ball_x, 2), "y": round(self.ball_y, 2),
                      "dx": round(self.ball_dx, 2), "dy": round(self.ball_dy, 2)},
            "your_paddle": {"x": round(paddle_x, 2), "width": self.PADDLE_W},
            "opponent_paddle": {"x": round(opponent_x, 2), "width": self.PADDLE_W},
            "board": {"width": self.WIDTH, "height": self.HEIGHT},
            "opponent_model": opponent_model,
            "opponent_prompt": opponent_prompt,
        }

    def move(self, side, action):
        change = -18 if action == LEFT else 18 if action == RIGHT else 0
        if side == "top":
            self.top_x = max(0, min(self.WIDTH - self.PADDLE_W, self.top_x + change))
        else:
            self.bottom_x = max(0, min(self.WIDTH - self.PADDLE_W, self.bottom_x + change))

    def step(self):
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        if self.ball_x <= 0 or self.ball_x >= self.WIDTH:
            self.ball_dx *= -1

        winner = None
        if self.ball_dy < 0 and self.ball_y <= self.PADDLE_H + self.BALL:
            if self.top_x - self.BALL <= self.ball_x <= self.top_x + self.PADDLE_W + self.BALL:
                self.ball_y = self.PADDLE_H + self.BALL
                self.ball_dy = abs(self.ball_dy)
            else:
                winner = "bottom"
        elif self.ball_dy > 0 and self.ball_y >= self.HEIGHT - self.PADDLE_H - self.BALL:
            if self.bottom_x - self.BALL <= self.ball_x <= self.bottom_x + self.PADDLE_W + self.BALL:
                self.ball_y = self.HEIGHT - self.PADDLE_H - self.BALL
                self.ball_dy = -abs(self.ball_dy)
            else:
                winner = "top"

        if winner:
            if winner == "top":
                self.top_score += 1
            else:
                self.bottom_score += 1
            self.ball_x, self.ball_y = self.WIDTH / 2, self.HEIGHT / 2
            self.ball_dy = 4.0 if winner == "top" else -4.0
        return winner


def parse_action(text):
    normalized = " ".join(str(text).strip().lower().split())
    return normalized if normalized in {LEFT, RIGHT, STAY} else STAY


async def ask_ai(ai_module, model, prompt, observation):
    messages = [
        ai_module.system_message(prompt),
        ai_module.user_message(json.dumps(observation, indent=2)),
    ]
    async with ai_module.Agent().run(model, messages) as stream:
        chunks = []
        async for event in stream:
            if isinstance(event, ai_module.events.TextDelta):
                chunks.append(event.chunk)
        return "".join(chunks).strip()


async def run_ai_agent():
    """Run two AI players with JSON observations and traceback diagnostics."""
    root = None
    try:
        import ai
        import tkinter as tk

        player_model_name = os.environ.get("AI_SDK_DEFAULT_MODEL", "openai/gpt-5.4")
        opponent_model_name = "anthropic:claude-sonnet-4-6"
        player_model = ai.get_model(player_model_name)
        opponent_model = ai.get_model(opponent_model_name)
        player_prompt = AI_PROMPT + " You are the bottom paddle."
        opponent_prompt = AI_PROMPT + " You are the top paddle."

        game = HorizontalPong()
        root = tk.Tk()
        root.title("Two-AI Pong")
        canvas = tk.Canvas(root, width=game.WIDTH, height=game.HEIGHT, bg="#101827")
        canvas.pack()
        label = tk.Label(root, text="Starting both AI players...")
        label.pack()
        closed = False

        def close():
            nonlocal closed
            closed = True
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", close)

        while not closed:
            root.update()
            player_observation = game.observation("bottom", opponent_model_name, opponent_prompt)
            opponent_observation = game.observation("top", player_model_name, player_prompt)
            results = await asyncio.gather(
                ask_ai(ai, player_model, player_prompt, player_observation),
                ask_ai(ai, opponent_model, opponent_prompt, opponent_observation),
                return_exceptions=True,
            )
            player_text = results[0] if isinstance(results[0], str) else ""
            opponent_text = results[1] if isinstance(results[1], str) else ""
            if not isinstance(results[0], str):
                print("Player AI error:")
                traceback.print_exception(results[0])
            if not isinstance(results[1], str):
                print("Opponent AI error:")
                traceback.print_exception(results[1])

            player_action, opponent_action = parse_action(player_text), parse_action(opponent_text)
            game.move("bottom", player_action)
            game.move("top", opponent_action)
            game.step()

            canvas.delete("all")
            canvas.create_rectangle(game.top_x, 25, game.top_x + game.PADDLE_W,
                                    25 + game.PADDLE_H, fill="#ff8a80")
            canvas.create_rectangle(game.bottom_x, game.HEIGHT - 25 - game.PADDLE_H,
                                    game.bottom_x + game.PADDLE_W, game.HEIGHT - 25,
                                    fill="#80d8ff")
            canvas.create_oval(game.ball_x - game.BALL / 2, game.ball_y - game.BALL / 2,
                               game.ball_x + game.BALL / 2, game.ball_y + game.BALL / 2,
                               fill="#ffffff")
            label.config(text=(f"Bottom {game.bottom_score} ({player_model_name}) - "
                               f"Top {game.top_score} ({opponent_model_name}) | "
                               f"Player: {player_action} | Opponent: {opponent_action}"))
            root.update()
            await asyncio.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping AI Pong...")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--agent":
        asyncio.run(run_ai_agent())
    else:
        run()


if __name__ == "__main__":
    main()
