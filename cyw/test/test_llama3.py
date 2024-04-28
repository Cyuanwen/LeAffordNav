# from langchain_community.llms import Ollama
# llm = Ollama(model="llama3:8b")
# llm.invoke("Why is the sky blue?")
# print('over')
# # 能够运行，但是没有输出

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='llama3:8b',
)

print('over')
