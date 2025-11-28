# src/utils.py

from dotenv import load_dotenv
import os

def setup_environment():
    """Carrega variáveis de ambiente e configura o LangSmith."""
    load_dotenv()
    
    # 1. Tenta carregar as chaves (prioriza GEMINI, fallback GOOGLE_API_KEY)
    gemini_key = os.getenv("GOOGLE_API_KEY") 

    # 2. DEBUG (Pode remover agora, mas deixamos para garantir)
    print(f"DEBUG: Chave carregada (Primeiros 5 dígitos): {gemini_key[:5] if gemini_key else 'NÃO CARREGADA'}")
    
    if not gemini_key:
        print("🚨 ERRO: GEMINI_API_KEY não configurada no .env.")
    
    # ... (o resto do código do LangSmith) ...
    
    return gemini_key