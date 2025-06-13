import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

config = {
    "chat": st.secrets["chat"],
    "embedding": st.secrets["embedding"]
}

embedder = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding"]["azure_endpoint"],
    azure_deployment=config["embedding"]["azure_deployment"],
    api_version=config["embedding"]["api_version"],
    api_key=config["embedding"]["azure_api_key"]
)

vector_store = InMemoryVectorStore(embedder)

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    api_version=config["chat"]["api_version"],
    api_key=config["chat"]["azure_api_key"],
)

def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        (
            "user",
            f"""Extract from the content the following metadata.
Answer 'unknown' if you cannot find or generate the information.
Metadata list:
- title
- author
- source
- type of content (e.g. scientific paper, litterature, news, etc.)
- language
- themes as a list of keywords

<content>
{extract}
</content>"""
        )
    ]
    response = llm.invoke(messages)
    return response.content

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)
    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
        }
    if use_meta_doc:
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(page_content=get_meta_doc(extract),
                            metadata={
                                'document_name': doc_name,
                                'insert_date': datetime.now()
                            })
        all_splits.append(meta_doc)
    _ = vector_store.add_documents(documents=all_splits)
    return

def delete_file_from_store(name: str) -> int:
    ids_to_remove = []
    for (id, doc) in vector_store.store.items():
        if name == doc['metadata']['document_name']:
            ids_to_remove.append(id)
    vector_store.delete(ids_to_remove)
    return len(ids_to_remove)

def retrieve(question: str, k: int = 5):
    return vector_store.similarity_search(question, k=k)

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
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, docs_content, language)
    response = llm.invoke(messages)
    return response.content

def inspect_vector_store(top_n: int = 5):
    for i, (doc_id, doc) in enumerate(vector_store.store.items()):
        if i >= top_n:
            break
        print(f"[{i}] {doc['metadata'].get('document_name')} : {doc['text'][:100]}")
