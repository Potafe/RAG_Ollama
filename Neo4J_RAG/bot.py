import os

import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

graph = Neo4jGraph(url=url, username=username, password=password)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

llm_transformer = LLMGraphTransformer(llm=llm)

def main():
    st.header("📄Chat with your pdf file")

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create a LangChain Document
        # document = Document(
        #     page_content=text,
        #     metadata={
        #         "file_name": pdf.name,
        #         "file_size": pdf.size,
        #         "num_pages": len(pdf_reader.pages),
        #     },
        # )

        # st.write("Document initialized successfully:", document)

        # # Wrap document in a list if expected by the function
        # try:
        #     graph_documents = llm_transformer.convert_to_graph_documents([document])
        #     for gd in graph_documents:
        #         st.write("Graph nodes:", gd.nodes)
        #         st.write("Graph relationships:", gd.relationships)
        # except AttributeError as e:
        #     st.error(f"Error converting document to graph: {e}")
        #     st.write("Document content debug:", document)

        text = """
        Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
        She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
        Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
        She was, in 1906, the first woman to become a professor at the University of Paris.
        """
        documents = [Document(page_content=text)]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        st.write(f"Nodes:{graph_documents[0].nodes}")
        st.write(f"Relationships:{graph_documents[0].relationships}")


        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,  # Delete existing PDF data
        )
        
        graph.add_graph_documents(graph_documents)
        
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")

        if query:
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])


if __name__ == "__main__":
    main()