import os
import re

import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from feedback import salvar_feedback, salvar_pergunta
from rag_utils import *
import logging
import warnings



def resetar_chat() -> None:
    st.session_state.messages = []

# Caso seja utilizado um modelo com capacidade de raciocinio [reasoning/think]
# def extrair_resposta(texto):
#     return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    st.set_page_config(
        page_title="📄 RAG com PDFs + Ollama",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    st.title("📄 Assitente de conversação para documentos")
    st.caption(
        "Faça upload dos seus PDFs e receba informações rápidas sobre o conteúdo, de forma segura"
        
    )
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        st.session_state.rag_chain = None


    with st.sidebar:
        st.header("📂 Documentos", divider="gray")
        arquivos = st.file_uploader(
            "Envie um ou mais PDFs:",
            type="pdf",
            accept_multiple_files=True,
        )

        ## Exibe uma lista com todos os arquivos já carregados
        # if arquivos:
        #     st.markdown("**PDFs carregados:**")
        #     for arquivo in arquivos:
        #         st.write(f"- {arquivo.name}")

        # Processa os arquivos e armazena no vector store
        if st.button("Processar PDFs", use_container_width=True) and arquivos:
            docs = carregar_documentos(arquivos)
            retriever, rag_chain = construir_rag(docs)
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.success("PDF(s) processado(s) com sucesso!")
        
        # Limpa a conversa para uma nova pergunta
        st.button("Limpar Conversa", on_click=resetar_chat, use_container_width=True)

        if st.button("🗑️ Deletar Base de Conhecimento", use_container_width=True):
            deletar_base_conhecimento()
            st.session_state.retriever = None
            st.session_state.rag_chain = None
            st.success("Base de conhecimento deletada com sucesso!")

        avaliacao_checkbox = st.checkbox("Feedback", value=False, key="feedback")

    
    with st.container():
        st.header("💬 Chat com o Assistente", divider="gray")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        pergunta = st.chat_input("Digite sua pergunta aqui...")

        salvar_pergunta(pergunta)

        if pergunta and st.session_state.retriever:
            st.session_state.messages.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.write(pergunta)
            with st.chat_message("ai"):
                st.write("Estou processando sua pergunta...")
                try:
                    resposta = executar_pergunta(
                        pergunta, st.session_state.retriever, st.session_state.rag_chain
                    )
                except Exception as e:
                    resposta = f"Erro ao processar pergunta: {e}"
                    
                st.markdown(resposta)
                st.session_state.messages.append({"role": "ai", "content": resposta})
    
    with st.container():
        st.empty()
    
    col_1, col_2 = st.columns([2, 2])
    

    if "feedback_enviado" not in st.session_state:
        st.session_state.feedback_enviado = False
    
    with col_1:
        st.empty()
    with col_2:
        if st.session_state.feedback and not st.session_state.feedback_enviado:
            with st.container():
                st.header("✍️ Feedback", divider="gray")
                st.write("Escolha uma opção (Positivo/Negativo):")
                avaliacao = str(st.feedback("thumbs"))
                usuario = st.text_input("Seu nome", "")
                mensagem = st.text_area("Sua mensagem", "")

                usuario = "anonimous" if not usuario else usuario

                if st.button("Enviar"):
                    if all([avaliacao, usuario, mensagem]):
                        salvar_feedback(usuario=usuario, mensagem=mensagem, avaliacao=avaliacao)
                        st.success("✅ Feedback salvo com sucesso!")
                        st.session_state.feedback_enviado = True
                        
                    else:
                        st.warning("⚠️ Preencha todos os campos antes de enviar.")

        elif st.session_state.feedback_enviado:
            st.success("✅ Obrigado pelo feedback!")
            # na próxima renderização, reseta para esconder formulário
            # st.session_state.feedback = False
            st.session_state.feedback_enviado = False
        
        else:
            st.empty()



    st.markdown("---")
    st.caption("Desenvolvido com ❤️ por Núcleo de Ciência de Dados • Unimed Blumenau")


if __name__ == "__main__":
    main()
