import os
import json
from datetime import datetime, timezone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
try:
    API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
    print("🔴 ERROR: The 'GEMINI_API_KEY' environment variable is not set.")
    exit()

HISTORY_FILE = 'conversation_history.json'
MODEL_NAME = 'gemini-2.5-flash'

# --- Helper functions (serialization and saving) ---
# These functions ensure history is in the correct JSON format.
def history_to_json_serializable(history):
    serializable_history = []
    for content in history:
        serializable_history.append({
            "role": content.role,
            "parts": [part.text for part in content.parts]
        })
    return serializable_history

def save_history(filename, history):
    full_log = {
        "model_used": MODEL_NAME,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "messages": history_to_json_serializable(history)
    }
    try:
        with open(filename, 'w') as f:
            json.dump(full_log, f, indent=2)
    except IOError as e:
        print(f"🔴 ERROR: Could not write to file {filename}. Details: {e}")

# --- (STEP 1) Function to load history from file ---
def load_history(filename):
    """
    This function reads the JSON file and extracts the 'messages' list.
    This list is the history that will be injected into the model.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            # It returns a list of dictionaries, e.g., [{'role': 'user', 'parts': ['Hello']}]
            return data.get("messages", [])
    except (FileNotFoundError, json.JSONDecodeError):
        # If no history exists, it returns an empty list.
        return []

# --- Main function to run the chat ---
def main():
    """
    Initializes the model and runs the main chat loop.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    # --- (STEP 2) LOAD THE HISTORY ---
    # We call load_history() to get the past conversation from the JSON file.
    # The variable 'raw_history' now holds our conversation context.
    print(f"🔄 Loading history from '{HISTORY_FILE}'...")
    raw_history = load_history(HISTORY_FILE)
    
    # --- (STEP 3) INJECT THE HISTORY ---
    # This is the crucial step.
    # We pass the 'raw_history' list directly into the 'start_chat' method.
    # The Gemini library automatically uses this history as the starting
    # context for the new conversation session.
    chat = model.start_chat(history=raw_history)
    
    if raw_history:
        print(f"✅ History loaded. Starting chat with context from previous session.")
    else:
        print("ⓘ No previous history found. Starting a new conversation.")

    print("--- 🤖 Gemini Chatbot Activated ---")
    print("Type 'exit' or 'quit' to end the session.")
    print("------------------------------------")

    # --- Main Chat Loop ---
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            print("\n🤖 Session ended. Goodbye!")
            break

        try:
            # When you call 'send_message', the model already knows the history
            # that was injected during 'start_chat'.
            response = chat.send_message(user_input)
            print(f"Gemini: {response.text}")

            # The new turn is automatically added to chat.history.
            # We save the *entire* updated history for the next session.
            save_history(HISTORY_FILE, chat.history)

        except Exception as e:
            print(f"\n🔴 An error occurred: {e}")
            break

if __name__ == "__main__":
    main()