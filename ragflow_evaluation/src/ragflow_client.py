"""
RAGFlow API Client for querying the RAG system
"""
import requests
import json
import time
from typing import Dict, Optional, List
from config import RAGFLOW_CONFIG, EVAL_SETTINGS


class RAGFlowClient:
    """Client for interacting with RAGFlow API"""

    def __init__(self, base_url: str = None, api_key: str = None, chat_id: str = None):
        """
        Initialize RAGFlow client

        Args:
            base_url: RAGFlow base URL (defaults to config)
            api_key: RAGFlow API key (defaults to config)
            chat_id: RAGFlow chat ID (defaults to config)
        """
        self.base_url = base_url or RAGFLOW_CONFIG["base_url"]
        self.api_key = api_key or RAGFLOW_CONFIG["api_key"]
        self.chat_id = chat_id or RAGFLOW_CONFIG["chat_id"]
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def query(
        self,
        question: str,
        model_name: str = "default",
        stream: bool = False,
        temperature: float = 0.1,
        max_retries: int = 3
    ) -> Dict:
        """
        Query RAGFlow with a question

        Args:
            question: User question
            model_name: Embedding model name (should match RAGFlow configuration)
            stream: Whether to stream response
            temperature: LLM temperature
            max_retries: Number of retry attempts

        Returns:
            Dict containing:
                - answer: Generated answer
                - contexts: Retrieved contexts (if available)
                - success: Whether query succeeded
                - error: Error message (if failed)
        """
        url = f"{self.base_url}/api/v1/chats_openai/{self.chat_id}/chat/completions"

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": question}],
            "stream": stream,
            "temperature": temperature
        }

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=EVAL_SETTINGS["timeout"]
                )
                response.raise_for_status()

                # Parse response
                response_data = response.json()

                # Extract answer
                answer = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

                # Try to extract contexts if available
                # Note: RAGFlow response format may vary - adjust as needed
                contexts = self._extract_contexts(response_data)

                return {
                    "answer": answer,
                    "contexts": contexts,
                    "success": True,
                    "error": None,
                    "raw_response": response_data
                }

            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if hasattr(response, 'text'):
                    error_msg += f"\nResponse: {response.text}"

                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    return {
                        "answer": None,
                        "contexts": [],
                        "success": False,
                        "error": error_msg,
                        "raw_response": None
                    }
            except Exception as e:
                return {
                    "answer": None,
                    "contexts": [],
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "raw_response": None
                }

    def _extract_contexts(self, response_data: Dict) -> List[str]:
        """
        Extract retrieved contexts from RAGFlow response

        Note: RAGFlow's response format may vary. Adjust this method based on actual format.
        Common locations:
        - response_data['contexts']
        - response_data['choices'][0]['message']['contexts']
        - response_data['metadata']['retrieved_chunks']
        """
        contexts = []

        # Try various possible locations
        # Option 1: Top-level contexts
        if 'contexts' in response_data:
            contexts = response_data['contexts']

        # Option 2: Within message
        elif 'choices' in response_data and len(response_data['choices']) > 0:
            message = response_data['choices'][0].get('message', {})
            if 'contexts' in message:
                contexts = message['contexts']

        # Option 3: In metadata
        elif 'metadata' in response_data:
            if 'retrieved_chunks' in response_data['metadata']:
                contexts = response_data['metadata']['retrieved_chunks']
            elif 'contexts' in response_data['metadata']:
                contexts = response_data['metadata']['contexts']

        # Ensure contexts is a list of strings
        if isinstance(contexts, list):
            return [str(ctx) if not isinstance(ctx, str) else ctx for ctx in contexts]
        elif isinstance(contexts, str):
            return [contexts]
        else:
            return []

    def test_connection(self) -> bool:
        """Test RAGFlow connection with a simple query"""
        try:
            result = self.query("Hello, this is a test query.", max_retries=1)
            return result["success"]
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
