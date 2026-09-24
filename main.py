"""Pong launcher with a separate asynchronous AI chat process.

Run ``python main.py`` for the localhost-neuron Pong controller.
Run ``python main.py --agent`` for the two-window AI Pong experiment.

In AI mode, main.py owns the Pong window and sends JSON ball locations to
chat.py over localhost process pipes. chat.py owns the chat window, calls both
models asynchronously, displays their messages, and returns JSON actions.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from brain import TinyBrain
from pong_game import PongGame

ACTIONS = {"up": 5001, "stay": 5002, "down": 5003}
PROJECT_DIR = Path(__file__).resolve().parent


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
    """Run the original localhost-neuron Pong mode with tkinter."""
    import tkinter as tk

    processes = start_neurons()
    root = None
    try:
        time.sleep(1.0)
        endpoints = {a: f"http://127.0.0.1:{p}" for a, p in ACTIONS.items()}
        brain, game = TinyBrain(endpoints), PongGame()
        if os.environ.get("HEADLESS") == "1":
            while True:
                observation = game.observation().as_dict()
                action, _ = brain.choose_action(observation)
                _, reward, done, _ = game.step(action)
                brain.learn(action, observation, reward)
                if done:
                    game.reset()

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
            action, _ = brain.choose_action(observation)
            _, reward, done, hit = game.step(action)
            brain.learn(action, observation, reward)
            state["reward"] += reward
            state["score"] += int(hit)
            canvas.delete("all")
            canvas.create_rectangle(game.paddle_x, game.paddle_y,
                                    game.paddle_x + game.PADDLE_W,
                                    game.paddle_y + game.PADDLE_H, fill="#eeeeee")
            canvas.create_oval(game.ball_x - 6, game.ball_y - 6,
                               game.ball_x + 6, game.ball_y + 6, fill="#50dc96")
            status.set(f"Episode {state['episode']} | Score {state['score']} | "
                       f"Reward {state['reward']:.1f} | Action: {action}")
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
            except Exception:
                pass
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


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

    def observation(self):
        return {
            "ball": {"x": round(self.ball_x, 2), "y": round(self.ball_y, 2),
                     "dx": round(self.ball_dx, 2), "dy": round(self.ball_dy, 2)},
            "top_paddle": {"x": round(self.top_x, 2), "width": self.PADDLE_W},
            "bottom_paddle": {"x": round(self.bottom_x, 2), "width": self.PADDLE_W},
            "board": {"width": self.WIDTH, "height": self.HEIGHT},
            "scores": {"top": self.top_score, "bottom": self.bottom_score},
        }

    def move(self, side, action):
        change = -18 if action == "left - CRL" else 18 if action == "right - LRC" else 0
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
        if winner == "top":
            self.top_score += 1
        elif winner == "bottom":
            self.bottom_score += 1
        if winner:
            self.ball_x, self.ball_y = self.WIDTH / 2, self.HEIGHT / 2
            self.ball_dy = 4.0 if winner == "top" else -4.0
        return winner


class ChatProcess:
    """Start chat.py and exchange newline-delimited JSON safely in a thread."""

    def __init__(self):
        self.output = queue.Queue()
        self.process = subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "chat.py")],
            cwd=str(PROJECT_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self):
        try:
            for line in self.process.stdout:
                try:
                    self.output.put(json.loads(line))
                except json.JSONDecodeError:
                    print(f"chat.py: {line.rstrip()}")
        except Exception:
            traceback.print_exc()

    def send(self, observation):
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.write(json.dumps(observation) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            traceback.print_exc()

    def poll(self):
        latest = None
        while True:
            try:
                latest = self.output.get_nowait()
            except queue.Empty:
                return latest

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def run_ai_agent():
    """Main Pong window; chat.py owns the separate AI conversation window."""
    import tkinter as tk

    game = HorizontalPong()
    chat = ChatProcess()
    root = tk.Tk()
    root.title("Two-AI Pong - Game")
    canvas = tk.Canvas(root, width=game.WIDTH, height=game.HEIGHT, bg="#101827")
    canvas.pack()
    status = tk.Label(root, text="Starting chat.py and both AI players...")
    status.pack()
    state = {"running": True, "last": None}

    def close():
        state["running"] = False
        chat.close()
        root.destroy()

    def tick():
        if not state["running"]:
            return
        observation = game.observation()
        # This is the alert carrying the current ball location to chat.py.
        chat.send({"type": "pong_observation", "observation": observation})
        result = chat.poll()
        if result:
            game.move("bottom", result.get("player_action", "stay"))
            game.move("top", result.get("opponent_action", "stay"))
            winner = game.step()
            state["last"] = result
            status.config(text=(f"Bottom {game.bottom_score} | Top {game.top_score} | "
                                f"Ball ({game.ball_x:.0f}, {game.ball_y:.0f}) | "
                                f"{result.get('player_action')} / {result.get('opponent_action')}"))
        canvas.delete("all")
        canvas.create_rectangle(game.top_x, 25, game.top_x + game.PADDLE_W,
                                25 + game.PADDLE_H, fill="#ff8a80")
        canvas.create_rectangle(game.bottom_x, game.HEIGHT - 25 - game.PADDLE_H,
                                game.bottom_x + game.PADDLE_W, game.HEIGHT - 25,
                                fill="#80d8ff")
        canvas.create_oval(game.ball_x - 6, game.ball_y - 6, game.ball_x + 6,
                           game.ball_y + 6, fill="#ffffff")
        root.after(40, tick)

    root.protocol("WM_DELETE_WINDOW", close)
    try:
        tick()
        root.mainloop()
    except Exception:
        traceback.print_exc()
        close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--agent":
        run_ai_agent()
    else:
        run()


if __name__ == "__main__":
    main()
