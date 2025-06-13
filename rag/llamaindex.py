from datetime import datetime
import streamlit as st

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

# ------------------- CONFIGURATION -----------------------

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

config = {
    "chat": st.secrets["chat"],
    "embedding": st.secrets["embedding"]
}

llm = AzureOpenAI(
    model="gpt-35-chat",
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["api_version"],
)

embedder = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["api_version"],
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()

# ------------------- FONCTIONS PRINCIPALES -----------------------

def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFReader()
    documents = loader.load(file_path)
    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_chunks)
        doc_idxs.extend([doc_idx] * len(cur_chunks))

    nodes = []
    for idx, chunk in enumerate(text_chunks):
        node = TextNode(text=chunk)
        node.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now().isoformat()
        }
        embedding = embedder.get_text_embedding(node.get_content(metadata_mode="all"))
        print(f"Chunk {idx + 1} embedding (first 5 dims): {embedding[:5]}")
        node.embedding = embedding
        nodes.append(node)

    print(f"{len(nodes)} chunks ajoutés pour le document '{doc_name}'")
    vector_store.add(nodes)

def delete_file_from_store(name: str) -> int:
    raise NotImplementedError("Delete is not implemented for LlamaIndex.")

def retrieve(question: str, k: int = 5):
    query_embedding = embedder.get_query_embedding(question)
    print(f"Embedding pour la question (first 5 dims): {query_embedding[:5]}")
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=k,
        mode="default"
    )
    result = vector_store.query(vector_store_query)

    if result is None or result.nodes is None:
        print(f"Aucun résultat trouvé pour la question : '{question}'")
        return []

    print(f"{len(result.nodes)} documents trouvés pour la question : '{question}'")
    return result.nodes

def build_qa_messages(question: str, context: str, language: str) -> list:
    instructions = {
        "français": "Réponds en français.",
        "anglais": "
