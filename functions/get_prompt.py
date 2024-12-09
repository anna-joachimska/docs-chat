from langchain.schema.messages import SystemMessage

def get_system_message(system_prompts, key=None):
    selected_prompt = SystemMessage(content="You are a friendly assistant.")
    for prompt in system_prompts:
        if key in prompt:
            selected_prompt = prompt[key]
            break
    return selected_prompt