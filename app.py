import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# 1. Page Configuration
st.set_page_config(page_title="Academic RAG Assistant", layout="wide")
st.title("📚 Research Paper AI Explorer")

# 2. Load Environment & Key
load_dotenv()

# 3. Load the RAG Engine (Optimized with higher safety)
@st.cache_resource
def init_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("academic_faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # 2026 PRO TIP: gemini-1.5-flash is often more stable for free tier than 2.0-flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # Lowering 'k' to 3 reduces the tokens sent, preventing quota hits
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain

qa_chain = init_rag()

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User Input with Rate Limit Protection
# We use a 'form' to prevent the app from firing requests every time you type
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Ask about your papers:")
    submit_button = st.form_submit_button(label='Send to Gemini')

if submit_button and user_input:
    # Add to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing 2,500 pages..."):
            try:
                # Force a 1-second 'breather' for the API
                time.sleep(1) 
                
                response = qa_chain.invoke(user_input)
                answer = response["result"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                if "429" in str(e):
                    st.error("🛑 Rate Limit Hit! Google's Free Tier needs a 30-second break. Please wait before asking again.")
                else:
                    st.error(f"Error: {e}")