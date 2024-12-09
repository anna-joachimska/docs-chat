from langchain_openai import ChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from functions.get_prompt import get_system_message
import gradio as gr
import json
from prompts import rag_prompts
from .document_processor import get_relevant_context


llm = ChatOpenAI(model='gpt-4o-mini')

system_prompts = [{"funny mode": SystemMessage(content="You are a funny assistant. Try always respond with joke.")},
                  {"IT mode": SystemMessage(content="You are a IT assistant. Answer only questions about IT.")},
                  {"harry potter mode": SystemMessage(content="You are a friendly assistant, but all your answers must be in Harry Potter style.")}]


def append_history_to_file(message, role, filename):
    with open(filename, "r") as f:
        history = json.load(f)
    history.append({"role": role, "content": message})

    with open(filename, "w") as f:
        json.dump(history, f)


def new_chat(chat_mode=get_system_message(system_prompts), filename=None):
    if not filename:
        content = get_system_message(system_prompts, chat_mode).content
        chatbot = gr.Chatbot(height=500, type="messages", value=[{ "role": "system",
                                                                   "content": content}])
    else:
        with open(f"conversation_history/{filename}", "r") as f:
            history = json.load(f)
        chatbot = gr.Chatbot(height=500, type="messages", value=history)

    return chatbot


def llm_response(message, history, system_prompt_choice):
    history_langchain_format = [get_system_message(system_prompts, system_prompt_choice)]
    if len(history) == 1:
        if len(message) > 20:
            chat_name = message[:9]
        else:
            chat_name = message
        with open(f"conversation_history/{chat_name}.json", "w") as f:
            json.dump([], f)
    else:
        for msg in history:
            if msg['role'] == "user":
                if len(msg['content']) > 20:
                    chat_name = msg['content'][:9]
                else:
                    chat_name = msg['content']
                break
    # Get relevant context from vector store
    relevant_docs = get_relevant_context(message)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
        elif msg['role'] == "system":
            history_langchain_format.append(SystemMessage(content=msg['content']))

    history_langchain_format.append(HumanMessage(content=rag_prompts.standard_rag_prompt(message, context)))

    llm_chat_response = llm.invoke(history_langchain_format)

    append_history_to_file(message, "user", f"conversation_history/{chat_name}.json")
    append_history_to_file(llm_chat_response.content, "assistant", f"conversation_history/{chat_name}.json")

    return llm_chat_response.content
