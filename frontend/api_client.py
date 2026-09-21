import os

import requests
from dotenv import load_dotenv

load_dotenv()


class ApiClientError(RuntimeError):
    """Raised when the backend cannot be reached or returns an error."""


class ApiClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("API_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ApiClientError("API_BASE_URL is not configured.")

    def health(self) -> dict:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            raise ApiClientError("The backend health check failed.") from error

    def query(self, question: str) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"question": question},
                timeout=130,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            raise ApiClientError("The query request failed or timed out.") from error
