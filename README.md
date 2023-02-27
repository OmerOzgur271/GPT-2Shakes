# GPT-2Shakes

After running sh file python script fine-tunes gpt2 and saves trained mode, creates Fastapi endpoint. All traning and endpoint predictions happen in local machine.

For answer need to get request url = "http://0.0.0.0:8000/generate_text/"

input_text = "To be or not to be"

send GET request with input text
response = requests.get(url, params={"prompt": input_text})

print(response.text)
