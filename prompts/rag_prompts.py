def standard_rag_prompt(message, context):
    prompt = f"""
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just sat that you don't know.
        User Message: {message}
        Context: {context}
        Answer:
    """

    return prompt