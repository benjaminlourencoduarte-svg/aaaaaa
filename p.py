"""A tiny Pong environment used by the localhost neuron experiment."""

from dataclasses import dataclass
import random


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
        # Values are normalized so every neuron receives a small, readable number.
        return Observation(
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.ball_dx / 4.0,
            self.ball_dy / 3.0,
            (self.paddle_y + self.PADDLE_H / 2) / self.HEIGHT,
        )

    def step(self, action):
        """Move the paddle and ball. Return (observation, reward, done, hit)."""
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
            # A successful rally gives the learner time to make more decisions.
            self.ball_x = self.WIDTH - self.BALL_SIZE
            self.ball_dx *= -1

        return self.observation(), reward, missed, hit
