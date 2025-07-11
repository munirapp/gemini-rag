import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma # Import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# --- Konfigurasi API Key ---
try:
    API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
    print("🔴 ERROR: 'GEMINI_API_KEY' environment variable not set.")
    exit()

def get_or_create_vector_store(pdf_path, persist_directory):
    """
    Loads a vector store from disk if it exists, 
    otherwise creates it from the PDF and saves it to disk.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

    # Check if the vector store already exists on disk
    if os.path.exists(persist_directory):
        print(f"✅ Loading existing vector store from '{persist_directory}'...")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vector_store
    else:
        print(f"🔄 Creating new vector store from '{pdf_path}'...")
        
        # Load and process the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create the vector store and persist it
        print(f"💾 Saving new vector store to '{persist_directory}'...")
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
        return vector_store

def main():
    pdf_input_file = input("📂 Enter the PDF file name (e.g., document.pdf): ").strip()
    pdf_file = f"./examples/{pdf_input_file}"
    if not os.path.exists(pdf_file):
        print(f"🔴 ERROR: File '{pdf_file}' not found on folder examples.")
        return

    # Define a directory to save/load the vector store
    persist_directory = f"./chroma_db_{os.path.basename(pdf_file)}"
    
    # Get the vector store (either load or create)
    vector_store = get_or_create_vector_store(pdf_file, persist_directory)

    if not vector_store:
        print("❌ Failed to start chat session.")
        return
        
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, google_api_key=API_KEY)

    print("\n--- 🤖 RAG Chatbot is Active ---")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 35)

    while True:
        user_question = input("You: ").strip()
        if not user_question:
            continue
        if user_question.lower() in ['exit', 'quit']:
            break

        # The retrieval process is now much faster on subsequent runs
        retrieved_docs = vector_store.similarity_search(user_question, k=5)

        # --- retrieval logging ---
        print("\n" + "="*60)
        print("🔍 [LOG] Similarity Search Founded Context:")
        print("="*60)
            
        if not retrieved_docs:
            print("There are no relevan context founded")
        else:
            # Iterasi melalui setiap dokumen yang ditemukan untuk mencetak detailnya
            for i, doc in enumerate(retrieved_docs):
                # Mengambil nomor halaman dari metadata
                source_page = doc.metadata.get('page', 'unknown')
                
                print(f"\n--- Chunk {i+1} (Source: Page {source_page}) ---")
                sanitized_content = doc.page_content.replace("\n", " ")
                print(sanitized_content)
                print("-" * (len(f"--- Chunk {i+1} (Source: Page {source_page}) ---")))
        print("\n" + "="*60)
        print("🧠 [LOG] Sending Context into model...")
        print("="*60 + "\n")
        # --- retrieval logging ---
        
        context_for_prompt = ""
        for doc in retrieved_docs:
            context_for_prompt += f"{doc.page_content}\n\n"

        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_for_prompt}\nQuestion: {user_question}"
        
        response = model.invoke(prompt)
        print("\nGemini:", response.content)
        print("-" * 35)

if __name__ == "__main__":
    main()