# deque is for appending and popping from both ends, makes it faster
# import json
# data = context
# print(json.dumps(data, ident=2))

from collections import deque
MAX_MEMORY = 10
conversation_history = deque(maxlen=MAX_MEMORY)

def build_prompt(user_input):
    context = []

    for prompts in conversation_history:
        context.append(f"User: {prompts["user"]}")
        context.append(f"Assistant: {prompts["assistant"]}")

    # Add user_input after iterations are done
    context.append(f"User input: {user_input}")
    return "\n".join(context) 
# after building the prompt 
# add it to the memory

def add_to_memory(user_input, assistant_responses):
    conversation_history.append({
        "user": user_input,
        "assistant": assistant_responses
    }) 