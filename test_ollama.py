import ollama

response = ollama.chat(model='llama3.2:3b', messages=[
    {'role': 'user', 'content': 'Say hello in one short sentence.'}
])

print(response['message']['content'])