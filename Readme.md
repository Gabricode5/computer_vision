étape 1 : créer un environnement virtuel Python
    python3 -m venv venv

    Si windows faire cette commande : Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    
    source venv/bin/activate

étape 2 : installer les dépendances
    pip install -r project_pose/requirement.txt

étape 3 : lancer Ollama (BOB, l'IA locale)
    ollama serve
    ollama run llama3.2:3b

étape 4 : lancer le backend FastAPI (dans un terminal séparé)
    source venv/bin/activate
    uvicorn project_pose.agent:app --host 127.0.0.1 --port 8000

étape 5 : lancer l'application Streamlit (dans un terminal séparé)
    source venv/bin/activate
    streamlit run project_pose/app.py
