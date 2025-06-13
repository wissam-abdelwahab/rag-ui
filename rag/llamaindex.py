# rag/llamaindex.py
from datetime import datetime
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

import streamlit as st

config = {
    "chat": {
        "azure_deployment": st.secrets["chat"]["azure_deployment"],
        "azure_api_key": st.secrets["chat"]["azure_api_key"],
        "azure_endpoint": st.secrets["chat"]["azure_endpoint"],
        "azure_api_version": st.secrets["chat"]["azure_api_version"]
    },
    "embedding": {
        "azure_deployment": st.secrets["embedding"]["azure_deployment"],
        "azure_api_key": st.secrets["embedding"]["azure_api_key"],
        "azure_endpoint": st.secrets["embedding"]["azure_endpoint"],
        "azure_api_version": st.secrets["embedding"]["azure_api_version"]
    }
}


llm = AzureOpenAI(
    model=config["chat"]["azure_deployment"],
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["azure_api_version"],
)

embedder = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()

def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFReader()
    documents = loader.load(file_path)
    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        node.embedding = embedder.get_text_embedding(node.get_content(metadata_mode="all"))
        nodes.append(node)

    vector_store.add(nodes)
    return

def delete_file_from_store(name: str) -> int:
    raise NotImplemented('function not implemented for Llamaindex')

def retrieve(question: str, k: int = 5):
    query_embedding = embedder.get_query_embedding(question)
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=k,
        mode="default"
    )
    query_result = vector_store.query(vector_store_query)
    return query_result.nodes


def build_qa_messages(question: str, context: str, language: str) -> list[str]:
    instructions = {
        "français": "Réponds en français.",
        "anglais": "Answer in English.",
        "espagnol": "Responde en español.",
        "allemand": "Antwort auf Deutsch."
    }
    system_instruction = instructions.get(language, "Answer in English.")
    return [
        ("system", "You are an assistant for question-answering tasks."),
        (
            "system",
            f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
{system_instruction}
{context}"""
        ),
        ("user", question),
    ]

def answer_question(question: str, language: str = "français", k: int = 5) -> str:
    docs = retrieve(question, k)
    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    messages = build_qa_messages(question, docs_content, language)
    response = llm.invoke(messages)
    return response.content
