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
    model=config["chat"]["azure_deployment"],
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
        node.embedding = embedder.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        nodes.append(node)

    print(f"{len(nodes)} chunks ajoutés pour le document '{doc_name}'")
    vector_store.add(nodes)

def delete_file_from_store(name: str) -> int:
    raise NotImplementedError("Delete is not implemented for LlamaIndex.")

def retrieve(question: str, k: int = 5):
    query_embedding = embedder.get_query_embedding(question)
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=k,
        mode="default"
    )
    result = vector_store.query(vector_store_query)
    return result.nodes

def build_qa_messages(question: str, context: str, language: str) -> list:
    instructions = {
        "français": "Réponds en français.",
        "anglais": "Answer in English.",
        "espagnol": "Responde en español.",
        "allemand": "Antwort auf Deutsch."
    }
    lang_instruction = instructions.get(language, "Answer in English.")
    return [
        ("system", "You are an assistant for question-answering tasks."),
        (
            "system",
            f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
{lang_instruction}
{context}"""
        ),
        ("user", question),
    ]

def answer_question(question: str, language: str = "français", k: int = 5) -> str:
    docs = retrieve(question, k)
    if not docs:
        return "Aucun document pertinent trouvé pour cette question."
    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    messages = build_qa_messages(question, docs_content, language)
    response = llm.invoke(messages)
    return response.content



