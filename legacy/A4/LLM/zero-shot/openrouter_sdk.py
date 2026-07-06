import requests

class OpenRouter:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.2-3b-instruct", title="openrouter-call"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": title,
            "Content-Type": "application/json"
        }

    def chat(self, prompt: str, temperature=0.3, max_tokens=500):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"OpenRouter Error {response.status_code}: {response.text}")
        
        return response.json()['choices'][0]['message']['content'].strip()
