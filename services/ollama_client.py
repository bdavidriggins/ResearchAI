import requests
import json

class OllamaClient:
    def __init__(self, api_url='http://localhost:11434/api/generate'):
        self.api_url = api_url
        self.model = "llama2-uncensored"  # Use the smaller model for testing

    def generate_response(self, prompt, context, max_tokens=1024, temperature=0.7, stream=False):
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "max_tokens": max_tokens,  # Adjust tokens for smaller models
            "temperature": temperature,
            "stream": stream
        }
        try:
            response = requests.post(self.api_url, json=payload, stream=stream)
            response.raise_for_status()

            if stream:
                result = ""
                for line in response.iter_lines():
                    if line:
                        result += json.loads(line.decode('utf-8')).get('response', '')
                return result
            else:
                return response.json().get('response', '')

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""
