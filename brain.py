import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
load_dotenv()



embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("academic_faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5})
)

print("\n--- Research Paper Assistant is Ready! ---")
while True:
    user_query = input("\nAsk a question about your papers (or type 'exit'): ")
    if user_query.lower() == 'exit':
        break
    
    print("Thinking...")
    response = qa_chain.invoke(user_query)
    print(f"\nAnswer: {response['result']}")