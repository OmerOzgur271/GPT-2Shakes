# GPT-2Shakes

For answer need to get request url = "http://0.0.0.0:8000/generate_text/"

# define input text
input_text = "To be or not to be"

# send GET request with input text
response = requests.get(url, params={"prompt": input_text})

# print generated text
print(response.text)
