import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Konfigurasi Streamlit
st.set_page_config(page_title="RAG PDF Reader", page_icon="📚", layout="wide")
st.title("📚 RAG PDF Reader dengan Gemini")
st.markdown("Upload file PDF dan ajukan pertanyaan tentang isinya!")

# Sidebar untuk konfigurasi
st.sidebar.header("⚙️ Konfigurasi")
api_key = st.sidebar.text_input("Google API Key", type="password", 
                               help="Masukkan API key Google Gemini Anda")

if not api_key:
    st.sidebar.warning("Silakan masukkan Google API Key untuk melanjutkan")
    st.stop()

# Set environment variable
os.environ["GOOGLE_API_KEY"] = api_key

# Fungsi untuk memproses PDF dan return text chunks
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Ekstrak teks dari PDF dan return chunks"""
    try:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Hapus file sementara
        os.unlink(tmp_file_path)
        
        # Return serializable data
        text_chunks = []
        for doc in texts:
            text_chunks.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return text_chunks
        
    except Exception as e:
        st.error(f"Error memproses PDF: {str(e)}")
        return []

# Fungsi untuk membuat vector store (tidak di-cache)
@st.cache_resource
def create_vectorstore(text_chunks, _api_key):
    """Membuat vector store dari text chunks"""
    try:
        if not text_chunks:
            return None
            
        # Recreate Document objects
        from langchain.schema import Document
        documents = []
        for chunk in text_chunks:
            documents.append(Document(
                page_content=chunk['content'],
                metadata=chunk['metadata']
            ))
        
        # Buat embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=_api_key
        )
        
        # Buat vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error membuat vector store: {str(e)}")
        return None

# Fungsi untuk setup RAG chain
@st.cache_resource
def setup_rag_chain(_vectorstore, _api_key):
    """Setup RAG chain dengan Gemini"""
    try:
        if not _vectorstore:
            return None
            
        # Inisialisasi model Gemini
        llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=_api_key,
            temperature=0.3
        )
        
        # Custom prompt template
        prompt_template = """
        Gunakan konteks berikut untuk menjawab pertanyaan. Jika Anda tidak tahu jawabannya berdasarkan konteks, katakan saja bahwa Anda tidak tahu.
        
        Konteks: {context}
        
        Pertanyaan: {question}
        
        Jawaban yang detail dan informatif:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Setup retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error setup RAG chain: {str(e)}")
        return None

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Pilih file PDF",
        type="pdf",
        help="Upload file PDF yang ingin Anda analisis"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File berhasil diupload: {uploaded_file.name}")
        
        # Ekstrak teks dari PDF
        with st.spinner("🔄 Mengekstrak teks dari PDF..."):
            text_chunks = extract_text_from_pdf(uploaded_file)
        
        if text_chunks:
            st.success(f"✅ Teks berhasil diekstrak! ({len(text_chunks)} chunks)")
            
            # Buat vector store
            with st.spinner("🔄 Membuat vector store..."):
                vectorstore = create_vectorstore(text_chunks, api_key)
            
            if vectorstore:
                st.success("✅ Vector store berhasil dibuat!")
                
                # Simpan vectorstore di session state
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_processed = True
            else:
                st.error("❌ Gagal membuat vector store")
        else:
            st.error("❌ Gagal mengekstrak teks dari PDF")

with col2:
    st.header("💬 Tanya Jawab")
    
    if 'pdf_processed' in st.session_state and st.session_state.pdf_processed:
        # Setup RAG chain
        if 'qa_chain' not in st.session_state:
            with st.spinner("🔄 Menyiapkan sistem RAG..."):
                st.session_state.qa_chain = setup_rag_chain(st.session_state.vectorstore, api_key)
        
        if st.session_state.qa_chain:
            # Input pertanyaan
            question = st.text_area(
                "Ajukan pertanyaan tentang PDF:",
                height=100,
                placeholder="Contoh: Apa poin utama dalam dokumen ini?"
            )
            
            if st.button("🔍 Cari Jawaban", type="primary"):
                if question:
                    with st.spinner("🤔 Mencari jawaban..."):
                        try:
                            result = st.session_state.qa_chain({"query": question})
                            
                            # Tampilkan jawaban
                            st.subheader("💡 Jawaban:")
                            st.write(result['result'])
                            
                            # Tampilkan sumber
                            if result.get('source_documents'):
                                st.subheader("📚 Sumber:")
                                for i, doc in enumerate(result['source_documents']):
                                    with st.expander(f"Sumber {i+1}"):
                                        st.write(doc.page_content)
                                        if hasattr(doc, 'metadata'):
                                            st.write(f"**Halaman:** {doc.metadata.get('page', 'N/A')}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("⚠️ Silakan masukkan pertanyaan")
        else:
            st.error("❌ Gagal menyiapkan sistem RAG")
    else:
        st.info("📄 Upload PDF terlebih dahulu untuk mulai bertanya")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Informasi")
st.sidebar.markdown("""
**Cara Penggunaan:**
1. Masukkan Google API Key
2. Upload file PDF
3. Tunggu proses selesai
4. Ajukan pertanyaan tentang PDF
5. Sistem akan memberikan jawaban berdasarkan isi PDF

**Fitur:**
- ✅ Membaca PDF multi-halaman
- ✅ Pencarian semantik
- ✅ Jawaban kontekstual
- ✅ Menampilkan sumber jawaban
""")

# Footer
st.markdown("---")
st.markdown("🚀 **RAG PDF Reader** - Powered by Google Gemini & LangChain")

# Script untuk menjalankan dari command line
if __name__ == "__main__":
    # Untuk menjalankan: streamlit run script_name.py
    pass
