"""Client-side tiny brain: localhost HTTP connections are its synapses."""

import traceback
import requests


class TinyBrain:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.timeout = 0.15

    def choose_action(self, observation):
        scores = {}
        for action, url in self.endpoints.items():
            try:
                response = requests.post(url + "/signal", json=observation, timeout=self.timeout)
                response.raise_for_status()
                scores[action] = float(response.json().get("score", 0.0))
            except requests.RequestException:
                # A dead neuron cannot freeze the game: it simply receives a neutral score.
                traceback.print_exc()
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
            traceback.print_exc()
