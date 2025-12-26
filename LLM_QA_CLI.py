#!/usr/bin/env python3
"""
LLM_QA_CLI.py
NLP Question-and-Answering System - Command Line Interface
Accepts natural-language questions, preprocesses them, and sends to an LLM API
"""

import os
import re
import string
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from nltk.tokenize import word_tokenize
from nltk import download as nltk_download

# Load environment variables
load_dotenv()

# NLTK setup
try:
    nltk_download('punkt', quiet=True)
except:
    pass

# Configuration
LLM_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # Options: gemini, openai, cohere, groq


class TextPreprocessor:
    """Handles text preprocessing: lowercasing, tokenization, punctuation removal"""
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def tokenize(text: str) -> list:
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except:
            # Fallback to simple split
            return text.split()
    
    @staticmethod
    def preprocess(text: str, lowercase=True, remove_punct=True, tokenize_text=False) -> str | list:
        """
        Apply preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            lowercase: Apply lowercasing
            remove_punct: Remove punctuation
            tokenize_text: Tokenize into words
            
        Returns:
            Preprocessed text or list of tokens
        """
        if lowercase:
            text = TextPreprocessor.lowercase(text)
        
        if remove_punct:
            text = TextPreprocessor.remove_punctuation(text)
        
        if tokenize_text:
            return TextPreprocessor.tokenize(text)
        
        return text


class LLMClient:
    """Handles communication with various LLM APIs"""
    
    def __init__(self, provider: str = "gemini", api_key: str = None, model: str = None):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider (gemini, openai, cohere, groq)
            api_key: API key for the provider
            model: Model name to use
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or "gemini-1.5-flash"
        self.max_retries = 3
        self.retry_delay = 1
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        if not self.api_key:
            return "ERROR: GEMINI_API_KEY not set"
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent".format(
            model=self.model
        )
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }
        
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    params={"key": self.api_key},
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 503 and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[Retry {attempt + 1}/{self.max_retries}] API overloaded. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and data["candidates"]:
                        return data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "No response")
                    return "ERROR: Invalid response format from API"
                
                return f"ERROR: API returned status {response.status_code}"
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[Retry {attempt + 1}/{self.max_retries}] Network error. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return f"ERROR: Network error - {str(e)}"
        
        return "ERROR: Max retries exceeded"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not self.api_key:
            return "ERROR: OPENAI_API_KEY not set"
        
        url = "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": self.model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return f"ERROR: API returned status {response.status_code}"
        except requests.RequestException as e:
            return f"ERROR: Network error - {str(e)}"
    
    def _call_cohere(self, prompt: str) -> str:
        """Call Cohere API"""
        if not self.api_key:
            return "ERROR: COHERE_API_KEY not set"
        
        url = "https://api.cohere.ai/v1/generate"
        
        payload = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.2,
            "num_generations": 1,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("generations", [{}])[0].get("text", "No response").strip()
            return f"ERROR: API returned status {response.status_code}"
        except requests.RequestException as e:
            return f"ERROR: Network error - {str(e)}"
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        if not self.api_key:
            return "ERROR: GROQ_API_KEY not set"
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        payload = {
            "model": self.model or "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return f"ERROR: API returned status {response.status_code}"
        except requests.RequestException as e:
            return f"ERROR: Network error - {str(e)}"
    
    def query(self, prompt: str) -> str:
        """
        Send query to LLM API based on configured provider
        
        Args:
            prompt: Question/prompt to send to LLM
            
        Returns:
            Response from LLM
        """
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "cohere":
            return self._call_cohere(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        else:
            return f"ERROR: Unknown provider '{self.provider}'"


def main():
    """Main CLI loop"""
    print("\n" + "="*60)
    print("NLP Question-and-Answering System (CLI)")
    print("="*60)
    print(f"Provider: {LLM_PROVIDER.upper()}")
    print(f"Model: {LLM_MODEL}")
    print("="*60 + "\n")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    llm_client = LLMClient(provider=LLM_PROVIDER, api_key=LLM_API_KEY, model=LLM_MODEL)
    
    while True:
        try:
            # Get user question
            print("\nEnter your question (or 'quit' to exit):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                print("Please enter a valid question.")
                continue
            
            # Preprocessing steps
            print("\n[PREPROCESSING]")
            print(f"Original: {user_input}")
            
            # Lowercase
            lowercased = preprocessor.lowercase(user_input)
            print(f"Lowercased: {lowercased}")
            
            # Remove punctuation
            no_punct = preprocessor.remove_punctuation(lowercased)
            print(f"No Punctuation: {no_punct}")
            
            # Tokenization
            tokens = preprocessor.tokenize(no_punct)
            print(f"Tokens: {tokens}")
            
            # Reconstruct for API
            processed_question = " ".join(tokens)
            print(f"Processed: {processed_question}")
            
            # Query LLM
            print("\n[QUERYING LLM]")
            print("Sending request to API...")
            answer = llm_client.query(processed_question)
            
            # Display result
            print("\n[ANSWER]")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"ERROR: {str(e)}")
            continue


if __name__ == "__main__":
    main()
