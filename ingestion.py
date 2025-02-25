import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME

from dotenv import load_dotenv

load_dotenv()

def ingest_docs() -> None:
    # Load all PDFs from directory
    loader = DirectoryLoader(
        "licheng-info",
        glob="**/*.pdf",  # Match all PDFs in directory and subdirectories
        loader_cls=PyPDFLoader,
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        raw_documents = loader.load()
        print(f"loaded {len(raw_documents)} documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["*****separate*****", "\n\n", "\n", " ", ""],
        )
        documents = text_splitter.split_documents(documents=raw_documents)
        print(f"Splitted into {len(documents)} chunks")

        print(f"Starting insert {len(documents)} to Pinecone")
        PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
        print("****** Successfully added to Pinecone vectorstore ******")

    except Exception as e:
        print(f"Error during document processing: {str(e)}")

if __name__ == "__main__":
    ingest_docs()
