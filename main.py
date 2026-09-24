"""Pong controller and two-AI Pong demonstration.

Run ``python main.py`` for the existing localhost-neuron Pong controller.
Run ``python main.py --agent`` for a two-AI horizontal Pong match.

The AI match sends JSON observations to two models. Each model must answer with
exactly ``left - CRL`` or ``right - LRC`` (``stay`` is accepted as a safe
fallback). Both prompts include the other model's name and prompt.
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
# Existing localhost-neuron Pong mode
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
    """Run the original local Flask-neuron Pong experiment."""
    processes = start_neurons()
    screen = None
    pygame = None
    try:
        time.sleep(1.0)
        brain = TinyBrain({a: f"http://127.0.0.1:{p}" for a, p in ACTIONS.items()})
        game = PongGame()
        if os.environ.get("HEADLESS") == "1":
            while True:
                observation = game.observation().as_dict()
                action, _ = brain.choose_action(observation)
                _, reward, done, _ = game.step(action)
                brain.learn(action, observation, reward)
                if done:
                    game.reset()
        else:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((game.WIDTH, game.HEIGHT))
            pygame.display.set_caption("Localhost Neuron Pong")
            clock = pygame.time.Clock()
            font = pygame.font.Font(None, 24)
            episode = hits = frames = 0
            total_reward = 0.0
            running = True
            while running:
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
                screen.fill((15, 20, 35))
                pygame.draw.rect(screen, (230, 230, 240),
                                 (game.paddle_x, game.paddle_y,
                                  game.PADDLE_W, game.PADDLE_H))
                pygame.draw.circle(screen, (80, 220, 150),
                                   (int(game.ball_x), int(game.ball_y)), game.BALL_SIZE // 2)
                text = f"Episode {episode}  Score {hits}  Reward {total_reward:.1f}"
                screen.blit(font.render(text, True, (240, 240, 240)), (10, 10))
                pygame.display.flip()
                clock.tick(60)
                if done:
                    episode += 1
                    hits = 0
                    total_reward = 0.0
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
        self.ball_dx = 5.0
        self.ball_dy = 4.0
        self.top_x = self.WIDTH / 2 - self.PADDLE_W / 2
        self.bottom_x = self.WIDTH / 2 - self.PADDLE_W / 2
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
        if action == LEFT:
            change = -18
        elif action == RIGHT:
            change = 18
        else:
            change = 0
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
    """Accept only the requested protocol; invalid model text becomes stay."""
    normalized = " ".join(str(text).strip().lower().split())
    if normalized == LEFT:
        return LEFT
    if normalized == RIGHT:
        return RIGHT
    if normalized == STAY:
        return STAY
    return STAY


async def ask_ai(ai_module, model, prompt, observation):
    """Send JSON to a model and collect its streamed text response."""
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
    """Run Pong with two visible, competing AI agents.

    The bottom player uses AI_SDK_DEFAULT_MODEL, while the top opponent always
    uses anthropic:claude-sonnet-4-6. Each receives JSON and can see the other
    model name and prompt in that JSON. Tracebacks are printed for diagnostics;
    a failed request safely becomes ``stay`` for that frame.
    """
    try:
        import ai
        player_model_name = os.environ.get("AI_SDK_DEFAULT_MODEL", "openai/gpt-5.4")
        opponent_model_name = "anthropic:claude-sonnet-4-6"
        player_model = ai.get_model(player_model_name)
        opponent_model = ai.get_model(opponent_model_name)

        player_prompt = AI_PROMPT + " You are the bottom paddle."
        opponent_prompt = AI_PROMPT + " You are the top paddle."
        game = HorizontalPong()
        import tkinter as tk
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

            player_action = parse_action(player_text)
            opponent_action = parse_action(opponent_text)
            game.move("bottom", player_action)
            game.move("top", opponent_action)
            winner = game.step()

            canvas.delete("all")
            canvas.create_rectangle(game.top_x, 25, game.top_x + game.PADDLE_W, 25 + game.PADDLE_H, fill="#ff8a80")
            canvas.create_rectangle(game.bottom_x, game.HEIGHT - 25 - game.PADDLE_H,
                                    game.bottom_x + game.PADDLE_W, game.HEIGHT - 25, fill="#80d8ff")
            canvas.create_oval(game.ball_x - game.BALL / 2, game.ball_y - game.BALL / 2,
                               game.ball_x + game.BALL / 2, game.ball_y + game.BALL / 2, fill="#ffffff")
            label.config(text=(f"Bottom {game.bottom_score} ({player_model_name})  -  "
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
        try:
            if not closed:
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
