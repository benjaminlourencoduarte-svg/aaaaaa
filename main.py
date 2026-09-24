"""Run the visual Pong experiment and its three local neuron services."""

import os
import subprocess
import sys
import time
import traceback

from brain import TinyBrain
from pong_game import PongGame

ACTIONS = {"up": 5001, "stay": 5002, "down": 5003}


def start_neurons():
    processes = []
    for action, port in ACTIONS.items():
        weight_file = os.path.join(".brain_weights", action + ".json")
        processes.append(subprocess.Popen([sys.executable, "neuron_service.py", action, str(port), weight_file]))
    return processes


def run():
    processes = start_neurons()
    try:
        time.sleep(1.0)  # Give Flask processes time to bind their localhost ports.
        brain = TinyBrain({action: f"http://127.0.0.1:{port}" for action, port in ACTIONS.items()})
        game = PongGame()
        episode, total_reward, hits, frames = 1, 0.0, 0, 0
        screen = None
        pygame = None
        if os.environ.get("HEADLESS") != "1":
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((game.WIDTH, game.HEIGHT))
            pygame.display.set_caption("Localhost Neuron Pong")
            clock = pygame.time.Clock()
            font = pygame.font.Font(None, 24)

        running = True
        while running:
            if screen:
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

            if screen:
                screen.fill((15, 20, 35))
                pygame.draw.rect(screen, (230, 230, 240), (game.paddle_x, game.paddle_y, game.PADDLE_W, game.PADDLE_H))
                pygame.draw.circle(screen, (80, 220, 150), (int(game.ball_x), int(game.ball_y)), game.BALL_SIZE // 2)
                info = f"Episode {episode}  Score {hits}  Reward {total_reward:.1f}  Weights learn after hits/misses"
                screen.blit(font.render(info, True, (240, 240, 240)), (10, 10))
                pygame.display.flip()
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
        if screen:
            pygame.quit()
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    run()
