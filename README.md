# Analyse de documents avec RAG

Ce projet propose une application Streamlit qui permet d'analyser des documents PDF en utilisant un modèle de langage couplé à un système de récupération d'information (RAG - Retrieval-Augmented Generation).

Fonctionnalités :

- Téléversement de fichiers PDF
- Sélection du framework d'indexation : LangChain ou LlamaIndex
- Choix de la langue de réponse (français, anglais, espagnol, allemand)
- Choix du nombre de documents utilisés pour répondre à la question
- Génération de réponse à partir des documents
- Feedback utilisateur (stocké dans une base SQLite)

Installation :

1. Cloner le dépôt :
   git clone https://github.com/mon-utilisateur/mon-depot.git
   cd mon-depot

2. Créer un environnement virtuel (optionnel mais recommandé) :
   python -m venv env
   source env/bin/activate     (ou env\Scripts\activate sous Windows)

3. Installer les dépendances :
   pip install -r requirements.txt

4. Créer un fichier secrets/config.yaml avec les identifiants Azure OpenAI :

   chat:
     azure_deployment: ...
     azure_api_key: ...
     azure_endpoint: ...
     azure_api_version: ...

   embedding:
     azure_deployment: ...
     azure_api_key: ...
     azure_endpoint: ...
     azure_api_version: ...

Lancement de l'application :

   streamlit run app.py

Déploiement :

L'application peut être déployée sur Streamlit Cloud. Il suffit d'y connecter ce dépôt et de spécifier app.py comme script principal.
