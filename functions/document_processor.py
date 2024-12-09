from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def process_pdf(file_path):
    # Load PDF
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)

    # Create/get vector store
    vectorstore = Chroma(
        collection_name="docschat",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="chroma_db"
    )

    # Add document to vector store
    vectorstore.add_documents(splits)

    return vectorstore

def delete_from_chroma(file_path):
    try:
        vectorstore = Chroma(
            collection_name="docschat",
            embedding_function=OpenAIEmbeddings(),
            persist_directory="chroma_db"
        )

        # Get chunk IDs based on document path metadata
        ids = vectorstore.get(where={"source": file_path})['ids']

        # Delete documents from Chroma DB
        vectorstore.delete(ids)

    except Exception as e:
        print(f"Error deleting from Chroma DB: {e}")

def get_relevant_context(query, k=3):
    vectorstore = Chroma(
        collection_name="docschat",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="chroma_db"
    )

    return vectorstore.similarity_search(query, k)