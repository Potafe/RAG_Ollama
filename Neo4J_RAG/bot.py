import os
import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from chains import load_embedding_model, load_llm
from dotenv import load_dotenv

load_dotenv(".env")

# Configuration
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "ollama")
llm_name = os.getenv("LLM")
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

# Initialize models
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

logger.info(f"Embedding dimension: {dimension}")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

def clean_previous_vectors(url, username, password):
    """Clean up previous vector embeddings safely"""
    try:
        # Create a temporary vector store just to execute the cleanup
        temp_store = Neo4jVector(
            embedding=embeddings,
            url=url,
            username=username,
            password=password,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
        )
        # Use the newer Neo4j syntax for deletion
        cleanup_query = """
        MATCH (n:PdfBotChunk)
        WITH n LIMIT 10000
        DETACH DELETE n
        """
        temp_store.query(cleanup_query)
        logger.info("Successfully cleaned previous vectors")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        # Continue even if cleanup fails
        pass

def main():
    st.header("📄Chat with your PDF file")

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        try:
            # Read PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Clean previous vectors before creating new ones
            clean_previous_vectors(url, username, password)

            # Create vector store
            try:
                vectorstore = Neo4jVector.from_texts(
                    chunks,
                    url=url,
                    username=username,
                    password=password,
                    embedding=embeddings,
                    index_name="pdf_bot",
                    node_label="PdfBotChunk",
                    pre_delete_collection=False,  # We handle deletion separately
                )
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return

            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            # Query interface
            query = st.text_input("Ask questions about your PDF file")
            if query:
                stream_handler = StreamHandler(st.empty())
                try:
                    qa.run(query, callbacks=[stream_handler])
                except Exception as e:
                    st.error(f"Error processing query: {e}")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main()