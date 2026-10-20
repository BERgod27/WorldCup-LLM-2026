import argparse
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def start_interactive_chat(model_name="qwen3.5:9b"):
    """
    Start an interactive terminal chat with the analyst, backed by Ollama.
    """
    print(f"⚽ Starting the sports data analyst (model: {model_name})...")
    print("Tip: type 'exit' or 'quit' to end the conversation.\n")
    
    # Initialise the model in chat mode
    # reasoning=False turns off qwen3's thinking mode, so <think> blocks don't
    # appear in the streamed answer (langchain-ollama's name for Ollama's "think").
    llm = ChatOllama(model=model_name, temperature=0.3, reasoning=False)
    
    # Chat prompt that carries the conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sports data analyst specialising in international football (Mundial LLM). "
                   "Answer clearly and concisely, always grounded in facts, tactics and football metrics."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    # Build the chain
    chain = prompt | llm | StrOutputParser()
    
    # Keep this session's history in memory
    chat_history = ChatMessageHistory()
    
    def get_session_history(session_id: str):
        return chat_history

    # Wrap the chain so history is injected and updated automatically
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'sair']:
                print("Analyst: Ending the session. 🏆")
                break
                
            if not user_input.strip():
                continue
                
            print("\nAnalyst: ", end="", flush=True)
            
            # Stream the answer so text appears as it is generated
            for chunk in chain_with_history.stream(
                {"question": user_input},
                config={"configurable": {"session_id": "mundial_session_1"}}
            ):
                print(chunk, end="", flush=True)
            print()
            
        except (KeyboardInterrupt, EOFError):
            print("\nAnalyst: Session interrupted. 🏆")
            break
        except Exception as e:
            print(f"\n[ERROR] Problem talking to Ollama: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mundial LLM - Interactive football analysis chat")
    parser.add_argument("--model", type=str, default="qwen3.5:9b", help="Ollama model name (e.g. qwen3.5:9b, llama3.1:8b, mistral)")
    args = parser.parse_args()
    
    start_interactive_chat(model_name=args.model)