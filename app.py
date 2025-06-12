import os
import tempfile
import streamlit as st
import pandas as pd
import sqlite3

# ==========================
# Choix du framework (Q4)
# ==========================
framework = st.radio(
    "Choisissez le framework d'indexation",
    ("langchain", "llamaindex"),
    horizontal=True
)

# Import dynamique du module backend
if framework == "langchain":
    import rag.langchain as rag_backend
else:
    import rag.llamaindex as rag_backend

# ==========================
# Config de la page
# ==========================
st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

# ==========================
# Initialisation base SQLite (Q6b)
# ==========================
def init_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            response TEXT,
            feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(question: str, response: str, feedback: str):
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedbacks (question, response, feedback)
        VALUES (?, ?, ?)
    """, (question, response, feedback))
    conn.commit()
    conn.close()

# ==========================
# Application principale
# ==========================
def main():
    init_db()  # Initialise la base au démarrage

    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA en les chargeant dans l'application. Puis posez toutes vos questions.")

    # Téléversement de fichiers
    uploaded_files = st.file_uploader(
        label="Déposez vos fichiers ici ou chargez-les",
        type=None,
        accept_multiple_files=True
    )

    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_in_kb:.2f}"
            })

            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, "temp.pdf")
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                rag_backend.store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)

        df = pd.DataFrame(file_info)
        st.table(df)

    # Suppression de fichiers
    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        try:
            rag_backend.delete_file_from_store(name)
        except NotImplementedError:
            st.warning(f"La suppression n'est pas encore supportée par le framework {framework}")

    # ==========================
    # Sélecteur de langue (Q3)
    # ==========================
    language = st.selectbox(
        "Langue de réponse",
        ["français", "anglais", "espagnol", "allemand"]
    )

    # ==========================
    # Sélecteur k (Q5)
    # ==========================
    k = st.slider(
        "Nombre de documents à utiliser pour répondre",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    # Champ de question
    question = st.text_input("Votre question ici")

    # Bouton "Analyser"
    if st.button("Analyser"):
        model_response = rag_backend.answer_question(question, language=language, k=k)
        st.text_area("Zone de texte, réponse du modèle", value=model_response, height=200)

        # ==========================
        # Feedback utilisateur (Q6b)
        # ==========================
        feedback = st.feedback("Cette réponse vous a-t-elle été utile ?")
        if feedback:
            save_feedback(question, model_response, feedback)
            st.success("Merci pour votre retour !")

    else:
        st.text_area("Zone de texte, réponse du modèle", value="", height=200)

if __name__ == "__main__":
    main()
