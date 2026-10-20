"""
Quick check that Ollama is running and the model answers.

Usage (from the repository root):
    python scripts/check_ollama.py
    python scripts/check_ollama.py --model qwen3.5:9b
"""

import argparse
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def check_ollama_connection(model_name="mundial-analyst"):
    """Send one short question to the local Ollama model and print the answer."""
    print(f"Testing connection to model: {model_name}...")
    
    try:
        # Ollama serves on localhost:11434 by default. Low temperature keeps the answer focused.
        # reasoning=False turns off qwen3's thinking mode (langchain-ollama's name for Ollama's "think").
        llm = ChatOllama(model=model_name, temperature=0.2, reasoning=False)
        
        prompt = PromptTemplate.from_template(
            "You are a sports data analyst. Answer concisely.\n"
            "Question: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        
        question = "Explain what the xG (expected goals) metric means in football, in 2 sentences at most."
        print(f"\nQuestion sent: '{question}'\n")
        
        answer = chain.invoke({"question": question})
        
        print("Response received:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        return True
        
    except Exception as e:
        print("\nCould not talk to Ollama.")
        print(f"Make sure Ollama is running and the model '{model_name}' is installed.")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the connection to Ollama.")
    parser.add_argument("--model", type=str, default="mundial-analyst", help="Ollama model name (e.g. mundial-analyst, qwen3.5:9b, llama3.1:8b)")
    args = parser.parse_args()
    
    check_ollama_connection(model_name=args.model)
