import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

# 1. Definições de Parâmetros
VECTOR_STORE_PATH = "./chroma_db"
PROTOCOL_FILE = "protocolo_sepse.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def initialize_protocol_retriever(api_key: str) -> BaseRetriever:
    """
    Inicializa o banco de dados vetorial (Chroma) e retorna o Retriever.
    """

    Embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}, # Força o uso da CPU no Mac
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- B: Tenta carregar o Vector Store existente ---
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH,
                embedding_function=Embeddings
            )
            print("✅ Banco de dados vetorial de Protocolos carregado com sucesso.")
            
        except Exception as e:
            # Caso a estrutura do DB tenha sido corrompida, tenta reindexar
            print(f"❌ Erro ao carregar ChromaDB: {e}. Reindexando...")
            vector_store = index_protocols(Embeddings)

    else:
        # --- C: Cria a indexação (O processo de RAG) ---
        vector_store = index_protocols(Embeddings)


    # --- D: Retorna o objeto Retriever ---
    return vector_store.as_retriever(search_kwargs={"k": 3}) 


def index_protocols(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Carrega o PDF, divide e cria a indexação no ChromaDB."""
    print("⏳ Banco de dados de Protocolos não encontrado. Iniciando indexação...")
    
    # 1. Carregamento do Documento
    if not os.path.exists(PROTOCOL_FILE):
        raise FileNotFoundError(f"❌ Erro: O arquivo de protocolo '{PROTOCOL_FILE}' não foi encontrado na raiz do projeto.")
        
    loader = PyPDFLoader(PROTOCOL_FILE)
    docs = loader.load()
    print(docs)
    
    # 2. Divisão (Splitting)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(docs)
    print(f"   Documento dividido em {len(splits)} chunks.")

    # 3. Criação e Armazenamento dos Vetores
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    return vector_store


# Função para simular o acesso a dados estruturados (Prontuário)
def fetch_patient_data(patient_id: str) -> str:
    """
    Simula a busca em um banco de dados de prontuários (Dados Estruturados). 
    Esta função será uma 'Tool' no nosso LangGraph.
    """
    
    if patient_id == "P12345":
        return (
            "Prontuário de P12345:\n"
            "  - Paciente masculino, 65 anos.\n"
            "  - Glicemia atual: 150 mg/dL.\n"
            "  - Alergias: Penicilina.\n"
            "  - Exames Pendentes: Hemocultura (coleta há 2 horas)."
        )
    elif patient_id == "P67890":
        return (
            "Prontuário de P67890:\n"
            "  - Paciente feminino, 40 anos.\n"
            "  - Histórico: Dor abdominal aguda.\n"
            "  - Conduta: Solicitado Ultrassom abdominal."
        )
    else:
        return f"Paciente {patient_id} não encontrado ou sem dados urgentes."