
import os
import logging
# ... (outros imports) ...

from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.utils import setup_environment
from src.rag_pipeline import initialize_protocol_retriever, fetch_patient_data
from src.llm_setup import initialize_llm

# 1. Configurar Ambiente e pegar a chave (para LLM/Embeddings)
api_key = setup_environment()

if api_key:
    # 1. Configurar o RAG (para busca de contexto)
    print("\nIniciando setup do RAG...")
    # Chamamos a função, mas evitamos o teste de invoke
    retriever = initialize_protocol_retriever(api_key) 

    # 2. Configurar a LLM (Llama 3 fine-tunado)
    print("\nIniciando LLM (Llama 3)...") # <--- ADICIONE ESTA LINHA
    llm = initialize_llm() # <--- ADICIONE ESTA LINHA

    # --- TESTE MINIMAL DE LANGCHAIN (RAG Chain) ---
    print("\n--- TESTE LANGCHAIN: RAG Chain ---")
    
    # Pergunta de teste: precisa ser algo que esteja no seu banco de dados
    query_protocol = "Qual é o manejo imediato para sepse?"
    
    # 1. Busca os documentos (RAG)
    retrieved_docs = retriever.invoke(query_protocol)
    # Formata o contexto dos documentos para a LLM
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"-> RAG: {len(retrieved_docs)} documentos recuperados.")

    # 2. Define a cadeia de geração (LLM + Prompt)
    print("-> Configurando Prompt e Chain.")

    prompt_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Você é um assistente médico especialista, treinado em protocolos internos.
    Sua função é usar APENAS o CONTEXTO de protocolo fornecido abaixo para responder de forma clara à PERGUNTA.
    Se a pergunta não for respondida pelo contexto, diga 'Não consegui encontrar o protocolo exato para essa conduta.'
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    CONTEXTO:
    {context}

    PERGUNTA: {question}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

    rag_chain = (
    ChatPromptTemplate.from_template(prompt_template)
    | llm 
    | StrOutputParser() 
)
    
    # 3. Executa a cadeia (LLM com o contexto)
    response = rag_chain.invoke({"context": context, "question": query_protocol})
    
    print(f"-> Geração concluída. (Início da resposta: {response[:50]}...)")
    
    print("\n--- RESPOSTA COMPLETA DO LLAMA 3 ---")
    print(response)
    print("-----------------------------------")
    
    # 4. Início da definição do LangGraph
    print("\n--- LangGraph: Ponto de Partida ---") 
    # Agora sim, você pode definir o LangGraph aqui
    
    # ... (O código do LangGraph que sugeri na resposta anterior virá AQUI) ...