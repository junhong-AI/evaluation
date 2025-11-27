"""
Custom RAG Client using Ollama for generation and OpenAI embeddings for retrieval
"""
import os
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from openai import OpenAI
import requests
import json


class CustomRAGClient:
    """Custom RAG client with Ollama LLM and OpenAI embeddings"""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b",
        embedding_model: str = "text-embedding-3-small",
        documents_path: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize Custom RAG client

        Args:
            ollama_base_url: Base URL for Ollama API
            ollama_model: Ollama model to use for generation
            embedding_model: OpenAI embedding model
            documents_path: Path to documents directory or file
            top_k: Number of documents to retrieve
        """
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model
        self.top_k = top_k

        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Document store
        self.documents = []
        self.document_embeddings = []

        # Load documents if path provided
        if documents_path:
            self.load_documents(documents_path)

    def load_documents(self, documents_path: str):
        """
        Load documents from a file or directory

        Args:
            documents_path: Path to documents (txt, json, or directory)
        """
        path = Path(documents_path)

        if path.is_file():
            if path.suffix == '.txt':
                self._load_text_file(path)
            elif path.suffix == '.json':
                self._load_json_file(path)
            else:
                print(f"Unsupported file type: {path.suffix}")
        elif path.is_dir():
            # Load all text files in directory
            for file in path.rglob('*.txt'):
                self._load_text_file(file)
            for file in path.rglob('*.json'):
                self._load_json_file(file)

        # Create embeddings for all documents
        if self.documents:
            print(f"Creating embeddings for {len(self.documents)} documents...")
            self._create_document_embeddings()
            print(f"✓ Embeddings created")

    def _load_text_file(self, file_path: Path):
        """Load and chunk a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Simple chunking by paragraphs
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        self.documents.extend(chunks)

    def _load_json_file(self, file_path: Path):
        """Load documents from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Assume JSON contains list of documents or dict with 'documents' key
        if isinstance(data, list):
            self.documents.extend([str(doc) for doc in data])
        elif isinstance(data, dict) and 'documents' in data:
            self.documents.extend([str(doc) for doc in data['documents']])

    def add_documents(self, documents: List[str]):
        """
        Add documents to the RAG system

        Args:
            documents: List of document strings
        """
        self.documents.extend(documents)
        self._create_document_embeddings()

    def _create_document_embeddings(self):
        """Create embeddings for all documents"""
        if not self.documents:
            return

        # Create embeddings in batches
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i+batch_size]
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        self.document_embeddings = np.array(all_embeddings)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return response.data[0].embedding

    def _retrieve_documents(self, query: str, top_k: int = None) -> List[str]:
        """
        Retrieve most relevant documents using cosine similarity

        Args:
            query: Query text
            top_k: Number of documents to retrieve (default: self.top_k)

        Returns:
            List of retrieved document texts
        """
        if not self.documents or len(self.document_embeddings) == 0:
            return []

        k = top_k or self.top_k

        # Get query embedding
        query_embedding = np.array(self._get_embedding(query))

        # Calculate cosine similarity
        similarities = np.dot(self.document_embeddings, query_embedding) / (
            np.linalg.norm(self.document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Return top documents
        return [self.documents[i] for i in top_indices]

    def _generate_with_ollama(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Dict:
        """
        Generate response using Ollama

        Args:
            prompt: Full prompt including context
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with response and metadata
        """
        url = f"{self.ollama_base_url}/api/generate"

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            return {
                "success": True,
                "response": result.get("response", ""),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

    def query(
        self,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        custom_prompt_template: Optional[str] = None
    ) -> Dict:
        """
        Query the RAG system

        Args:
            question: User question
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            custom_prompt_template: Optional custom prompt template with {context} and {question} placeholders

        Returns:
            Dict containing:
                - answer: Generated answer
                - contexts: Retrieved contexts
                - success: Whether query succeeded
                - error: Error message (if failed)
        """
        # Retrieve relevant documents
        contexts = self._retrieve_documents(question)

        if not contexts:
            return {
                "answer": "No relevant documents found.",
                "contexts": [],
                "success": False,
                "error": "No documents in knowledge base"
            }

        # Build prompt
        if custom_prompt_template:
            prompt = custom_prompt_template.format(
                context="\n\n".join(contexts),
                question=question
            )
        else:
            # Default RAG prompt
            context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
            prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context_text}

Question: {question}

Answer the question based on the context above. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate.

Answer:"""

        # Generate response
        result = self._generate_with_ollama(prompt, temperature, max_tokens)

        if result["success"]:
            return {
                "answer": result["response"],
                "contexts": contexts,
                "success": True,
                "error": None
            }
        else:
            return {
                "answer": None,
                "contexts": contexts,
                "success": False,
                "error": result["error"]
            }

    def test_connection(self) -> bool:
        """Test connection to Ollama"""
        try:
            url = f"{self.ollama_base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self.ollama_model in model_names or any(self.ollama_model in name for name in model_names):
                return True
            else:
                print(f"Warning: Model '{self.ollama_model}' not found in Ollama")
                print(f"Available models: {model_names}")
                return False
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
