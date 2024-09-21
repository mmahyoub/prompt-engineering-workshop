'''
Illustration Example for Power of Prompt Engineering (Especially RAG)
'''
import os 
import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env variables 
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("api_key")


def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if pdf is not None:
        # Get text data
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Vectorstore
        vs = create_vectorstore(text)

        # User input 
        if prompt := st.chat_input("Ask a question about your PDF."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(get_response(prompt, vs))

def create_vectorstore(text):
       
    # Chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)

    # Indexing (Vector Store)
    vs = FAISS.from_texts(splits, embedding=OpenAIEmbeddings())

    return vs


# Retrieve and generate using the relevant snippets of the blog.
def get_response(query, vs):
  '''
  Inputs:
    - query: user question 
    - vs: vectorstore
  '''
  llm = ChatOpenAI()
  retriever = vs.as_retriever()

  # RAG prompt template
  prompt = hub.pull("rlm/rag-prompt")

  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)


  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  response = rag_chain.invoke(query)
  return response



if __name__ == "__main__":
    main()