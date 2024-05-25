
import os
import pandas as pd
import streamlit as st
import dill as pickle
import threading
import time
import langchain
import faiss
import pickle
import numpy as np
from langchain_openai import OpenAI
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from openai import OpenAI as OpenAIClient
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain


# Set the OpenAI API key
openai_api_key = "sk-proj-XXXX"  # Replace with OpenAI API key
os.environ['OPENAI_API_KEY'] = openai_api_key

st.title("Financial News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
docs_file_path = "docs.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAIClient(api_key=openai_api_key)

# Function to get embedding
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

if process_url_clicked:
    if urls:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Save documents to a pickle file for later retrieval
        with open(docs_file_path, "wb") as f:
            pickle.dump(docs, f)

        # Create embeddings and save them to FAISS index
        st.write("Generating embeddings...")
        df = pd.DataFrame({'combined': [doc.page_content for doc in docs]})
        df['ada_embedding'] = df['combined'].apply(lambda x: get_embedding(x.strip(), model='text-embedding-3-small'))

        # Convert embeddings to a numpy array
        embeddings_np = np.array(df['ada_embedding'].tolist()).astype('float32')

        # Create a FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        index.add(embeddings_np)  # Add embeddings to the index
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(index, f)
    else:
        st.error("Please enter at least one URL.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path) and os.path.exists(docs_file_path):
        with open(file_path, "rb") as f:
            index = pickle.load(f)
        with open(docs_file_path, "rb") as f:
            docs = pickle.load(f)

        # Function to search the FAISS index
        def search_faiss(query, k=5):
            query_embedding = np.array([get_embedding(query)]).astype('float32')
            D, I = index.search(query_embedding, k)
            return D, I

        # Perform search
        distances, indices = search_faiss(query)
        
        # Retrieve documents based on indices
        retrieved_docs = [docs[i] for i in indices[0]]

        # Create a retriever with the retrieved documents
        retriever = FAISS.from_documents(retrieved_docs, OpenAIEmbeddings())
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the result
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
    else:
        st.error("Please create the FAISS index first.")