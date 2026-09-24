"""Asynchronous AI chat worker for the two-AI Pong experiment.

chat.py owns the second window. It reads Pong observations as JSON lines from
main.py, sends both model requests concurrently with asyncio, displays their
messages, and writes one JSON action result per line back to main.py.
"""

import asyncio
import json
import os
import sys
import traceback

import tkinter as tk

LEFT = "left - CRL"
RIGHT = "right - LRC"
STAY = "stay"
PLAYER_PROMPT = (
    "You are the bottom Pong paddle. Reply with exactly left - CRL, right - LRC, "
    "or stay. Use the JSON ball location and paddle locations."
)
OPPONENT_PROMPT = (
    "You are the top Pong paddle. Reply with exactly left - CRL, right - LRC, "
    "or stay. Use the JSON ball location and paddle locations."
)


def parse_action(value):
    value = " ".join(str(value).strip().lower().split())
    return value if value in {LEFT, RIGHT, STAY} else STAY


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
        player_text = "AI error: staying still"
        player_result = ""
    else:
        player_text = player_result
    if isinstance(opponent_result, Exception):
        print("Opponent AI traceback:", file=sys.stderr)
        traceback.print_exception(opponent_result)
        opponent_text = "AI error: staying still"
        opponent_result = ""
    else:
        opponent_text = opponent_result

    return {
        "type": "ai_result",
        "player_action": parse_action(player_result),
        "opponent_action": parse_action(opponent_result),
        "player_message": player_text,
        "opponent_message": opponent_text,
        "ball": observation.get("ball", {}),
    }


async def worker(ai_module, player_model, opponent_model, result_queue):
    # asyncio.to_thread keeps blocking stdin reads away from the AI event loop.
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return
        try:
            payload = json.loads(line)
            result = await process_observation(ai_module, player_model, opponent_model, payload)
            await asyncio.to_thread(result_queue.put, result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            await asyncio.to_thread(result_queue.put, {
                "type": "ai_result", "player_action": STAY,
                "opponent_action": STAY, "player_message": str(exc),
                "opponent_message": str(exc),
            })


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
    queue_for_gui = __import__("queue").Queue()

    def append(message):
        text.config(state="normal")
        text.insert("end", message + "\n\n")
        text.see("end")
        text.config(state="disabled")

    def drain():
        while True:
            try:
                result = queue_for_gui.get_nowait()
            except __import__("queue").Empty:
                break
            append(f"Ball: {result.get('ball', {})}\n"
                   f"Player AI: {result.get('player_message', '')}\n"
                   f"Opponent AI: {result.get('opponent_message', '')}")
        root.after(50, drain)

    async def run_async():
        await worker(ai, player_model, opponent_model, queue_for_gui)

    def start_loop():
        try:
            asyncio.run(run_async())
        except Exception:
            traceback.print_exc(file=sys.stderr)

    import threading
    threading.Thread(target=start_loop, daemon=True).start()
    drain()
    root.mainloop()


if __name__ == "__main__":
    main()
