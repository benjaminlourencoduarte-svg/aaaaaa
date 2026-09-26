"""Asynchronous AI chat worker for the two-AI Pong experiment.

chat.py owns the second window and communicates with main.py over stdin/stdout
with newline-delimited JSON. It receives the current Pong state, asks the two
models asynchronously, prints the model text and actions to its own tkinter
window, and sends the chosen actions back to main.py.
"""

import asyncio
import json
import os
import queue
import sys
import threading
import traceback

import tkinter as tk

LEFT = "left - CRL"
RIGHT = "right - LRC"
STAY = "stay"
PLAYER_PROMPT = (
    "You are the bottom Pong paddle. Reply with exactly left - CRL, right - LRC, "
    "or stay. Use the JSON ball, paddle, and score state."
)
OPPONENT_PROMPT = (
    "You are the top Pong paddle. Reply with exactly left - CRL, right - LRC, "
    "or stay. Use the JSON ball, paddle, and score state."
)


def parse_action(value):
    value = " ".join(str(value).strip().lower().split())
    if value == LEFT.lower():
        return LEFT
    if value == RIGHT.lower():
        return RIGHT
    if value == STAY.lower():
        return STAY
    return STAY


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


async def process_observation(ai_module, player_model, opponent_model, payload):
    observation = payload.get("observation", payload)
    player_result, opponent_result = await asyncio.gather(
        ask_ai(ai_module, player_model, PLAYER_PROMPT, observation),
        ask_ai(ai_module, opponent_model, OPPONENT_PROMPT, observation),
        return_exceptions=True,
    )

    if isinstance(player_result, Exception):
        print("Player AI traceback:", file=sys.stderr)
        traceback.print_exception(player_result)
        player_message = "AI error: staying still"
        player_text = STAY
    else:
        player_message = player_result
        player_text = parse_action(player_result)

    if isinstance(opponent_result, Exception):
        print("Opponent AI traceback:", file=sys.stderr)
        traceback.print_exception(opponent_result)
        opponent_message = "AI error: staying still"
        opponent_text = STAY
    else:
        opponent_message = opponent_result
        opponent_text = parse_action(opponent_result)

    return {
        "type": "ai_result",
        "player_action": player_text,
        "opponent_action": opponent_text,
        "player_message": player_message,
        "opponent_message": opponent_message,
        "ball": observation.get("ball", {}),
    }


async def worker(ai_module, player_model, opponent_model, gui_queue):
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return
        try:
            payload = json.loads(line)
            result = await process_observation(ai_module, player_model, opponent_model, payload)
            gui_queue.put(result)
            print(json.dumps(result), flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            fallback = {
                "type": "ai_result",
                "player_action": STAY,
                "opponent_action": STAY,
                "player_message": "error",
                "opponent_message": "error",
                "ball": {},
            }
            gui_queue.put(fallback)
            print(json.dumps(fallback), flush=True)


def main():
    import ai

    player_model_name = os.environ.get("AI_SDK_DEFAULT_MODEL", "openai/gpt-5.4")
    opponent_model_name = "anthropic:claude-sonnet-4-6"
    player_model = ai.get_model(player_model_name)
    opponent_model = ai.get_model(opponent_model_name)

    root = tk.Tk()
    root.title("AI Chat - Pong Agents")
    text = tk.Text(root, width=110, height=24, state="disabled", wrap="word")
    text.pack()
    gui_queue = queue.Queue()

    def append(message):
        text.config(state="normal")
        text.insert("end", message + "\n\n")
        text.see("end")
        text.config(state="disabled")

    def drain():
        while True:
            try:
                result = gui_queue.get_nowait()
            except queue.Empty:
                break
            append(
                f"Ball: {result.get('ball', {})}\n"
                f"Player AI: {result.get('player_message', '')}\n"
                f"Opponent AI: {result.get('opponent_message', '')}\n"
                f"Actions: player={result.get('player_action', STAY)}, "
                f"opponent={result.get('opponent_action', STAY)}"
            )
        root.after(50, drain)

    async def run_async():
        await worker(ai, player_model, opponent_model, gui_queue)

    threading.Thread(
        target=lambda: asyncio.run(run_async()),
        daemon=True,
    ).start()

    drain()
    root.mainloop()


if __name__ == "__main__":
    main()
