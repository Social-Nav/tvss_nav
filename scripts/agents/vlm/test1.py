# import ollama
# response = ollama.chat(model='llava:latest', messages=[
#   {
#     'role': 'user',
#     'content': 'Who are you?',
#   },
# ])
# print(response['message']['content'])

import ollama

# stream = ollama.chat(
#     model='llava:latest',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

ollama.embed(model='llava:latest', input='The sky is blue because of rayleigh scattering')