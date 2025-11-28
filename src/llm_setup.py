# src/llm_setup.py

import os
# Adicionar esta nova importação:
from huggingface_hub import hf_hub_download 
from langchain_community.llms import LlamaCpp
from langchain_core.language_models import BaseLLM

# ... (outros imports de llama_cpp não são mais necessários) ...

# Informações do modelo fine-tunado
REPO_ID = "amandanespoli/llama-3-8b-bnb-4bit-perguntas-respostas-medicina"
FILENAME = "llama-3-8b.Q4_K_M.gguf"
MODELS_DIR = "./models" 
MODEL_PATH = os.path.join(MODELS_DIR, FILENAME) 

def initialize_llm() -> BaseLLM:
    """Inicializa a LLM Llama 3, garantindo que o GGUF esteja no local correto."""
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    if not os.path.exists(MODEL_PATH):
        print(f"⏳ Baixando o modelo Llama 3: {FILENAME} do Hugging Face. Isso pode levar alguns minutos...")
        
        # --- NOVO CÓDIGO DE DOWNLOAD ---
        # A função hf_hub_download garante que o arquivo vá para o 'local_dir'
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODELS_DIR,  # Força o download para a pasta ./models
            force_download=True    # Garante que ele substitua o arquivo existente (opcional)
        )
        # --- FIM NOVO CÓDIGO DE DOWNLOAD ---
        
        print("✅ Download concluído. Inicializando LLM...")
    else:
        print("✅ Modelo Llama 3 GGUF encontrado localmente. Inicializando LLM...")
        
    
    # 3. Inicializa o LlamaCpp do LangChain (com o caminho correto)
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.7,
        max_tokens=2048,
        n_ctx=4096, 
        verbose=False, 
    )
    
    return llm