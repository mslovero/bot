#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ======================
# CONFIGURACIÓN PRINCIPAL
# ======================
CONFIG = {
    "model_name": "muniBrown",
    "document_path": os.path.expanduser("~/Escritorio/proyectos/miNuevoRag/docs/causa-comun.txt"),
    "faiss_path": os.path.expanduser("~/Escritorio/proyectos/miNuevoRag/faiss_index_muniBrown"),
    "chunk_size": 300,
    "chunk_overlap": 75,
    "temperature": 0.1,
    "top_p": 0.8,
    "k_retrievals": 2,
}

load_dotenv()

# ===================
# INICIALIZACIÓN
# ===================
def initialize():
    """Verifica configuraciones básicas"""
    print("\n" + "=" * 50)
    print(f"Inicializando sistema RAG con modelo {CONFIG['model_name']}")
    print("=" * 50)

    if not os.path.exists(CONFIG["document_path"]):
        print(f"\n❌ Error: Archivo no encontrado en {CONFIG['document_path']}")
        exit()

# ===================
# PROCESAMIENTO DE DOCUMENTOS
# ===================
def process_documents():
    """Carga y procesa el documento"""
    print("\n📂 Procesando documento...")
    encodings = ['utf-8', 'latin-1']

    for encoding in encodings:
        try:
            loader = TextLoader(CONFIG["document_path"], encoding=encoding)
            documents = loader.load()
            print(f"\n✅ Documento cargado correctamente con encoding {encoding}")
            return documents
        except Exception as e:
            print(f"❌ Intento con encoding {encoding} falló: {str(e)}")

    print("\n⚠️ No se pudo cargar el archivo con ninguno de los encodings probados")
    exit()

# ===================
# BASE DE CONOCIMIENTO VECTORIAL
# ===================
def setup_vectorstore(documents):
    """Configura la base de datos vectorial"""
    print("\n🧠 Configurando base de conocimientos...")
    text_splitter = CharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"])
    texts = text_splitter.split_documents(documents)
    print(f"📝 Documento dividido en {len(texts)} fragmentos")

    try:
        embeddings = OllamaEmbeddings(model=CONFIG["model_name"])
        if os.path.exists(CONFIG["faiss_path"]):
            print(f"🔍 Cargando base vectorial existente de {CONFIG['faiss_path']}")
            db = FAISS.load_local(CONFIG["faiss_path"], embeddings, allow_dangerous_deserialization=True)
        else:
            print("🆕 Creando nueva base vectorial")
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(CONFIG["faiss_path"])
        return db
    except Exception as e:
        print(f"\n❌ Error al configurar la base vectorial: {str(e)}")
        exit()

# ===================
# SISTEMA RAG
# ===================
def setup_rag_system(db):
    """Configura la cadena RAG completa"""
    print("\n🔗 Configurando sistema RAG...")
    try:
        llm = Ollama(model=CONFIG["model_name"], temperature=CONFIG["temperature"], top_p=CONFIG["top_p"])
        retriever = db.as_retriever(search_kwargs={"k": CONFIG["k_retrievals"]})
        prompt_template = """
        Usando la información de los fragmentos, respondé a la pregunta de forma breve y precisa.
        Fragmentos relevantes:
        {context}
        Pregunta: {question}
        Respuesta literal:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    except Exception as e:
        print(f"\n❌ Error al configurar RAG: {str(e)}")
        exit()
