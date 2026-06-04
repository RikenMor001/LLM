# deque is for appending and popping from both ends, makes it faster
from collections import deque

MAX_MEMORY = 10
conversation_history = deque(maxlen=MAX_MEMORY)

def build_prompt(user_input):
    context = []

    for prompts in conversation_history:
        context.append(f"User: {prompts["user"]}")
        context.append(f"Assistant: {prompts["Assistant"]}")

    # Add user_input after iterations are done
    context.append(f"User input: {user_input}")
    return "\n".join(context)