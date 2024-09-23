import requests
import json
import logging
from typing import Optional

class OllamaClient:
    """
    A client for interacting with the Ollama API to generate responses based on a given prompt and context.
    """

    def __init__(
        self,
        api_url: str = 'http://localhost:11434/api/generate',
        model: str = 'llama2-uncensored'
    ):
        """
        Initialize the OllamaClient.

        Args:
            api_url (str): The URL of the Ollama API endpoint.
            model (str): The name of the model to use for generating responses.
        """
        self.api_url = api_url
        self.model = model
        self.session = requests.Session()

        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def generate_response(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Optional[str]:
        """
        Generate a response from the Ollama API based on the given prompt and context.

        Args:
            prompt (str): The user's question or prompt.
            context (str): Additional context to provide to the model.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Sampling temperature for response variability.
            stream (bool): Whether to stream the response.

        Returns:
            Optional[str]: The generated response, or None if an error occurred.
        """
        # Validate inputs
        if not isinstance(prompt, str) or not isinstance(context, str):
            self.logger.error("Prompt and context must be strings.")
            return None

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            self.logger.error("max_tokens must be a positive integer.")
            return None

        if not (0.0 <= temperature <= 1.0):
            self.logger.error("temperature must be between 0.0 and 1.0.")
            return None

        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                stream=stream,
                timeout=30
            )
            response.raise_for_status()

            if stream:
                return self._parse_stream_response(response)
            else:
                return self._parse_response(response)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None

    def _parse_response(self, response: requests.Response) -> Optional[str]:
        """
        Parse a non-streaming response from the Ollama API.

        Args:
            response (requests.Response): The HTTP response object.

        Returns:
            Optional[str]: The extracted response text, or None if parsing fails.
        """
        try:
            response_json = response.json()
            return response_json.get('response', '').strip()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error: {e}")
            return None

    def _parse_stream_response(self, response: requests.Response) -> Optional[str]:
        """
        Parse a streaming response from the Ollama API.

        Args:
            response (requests.Response): The HTTP response object.

        Returns:
            Optional[str]: The concatenated response text, or None if parsing fails.
        """
        result = ""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        line_content = json.loads(line.decode('utf-8'))
                        result += line_content.get('response', '')
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decoding error in stream: {e}")
                        continue
            return result.strip()
        except Exception as e:
            self.logger.error(f"Error reading streaming response: {e}")
            return None
