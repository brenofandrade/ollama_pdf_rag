
import logging
import warnings
import tempfile
from pathlib import Path
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MODEL_EMBEDDING = "mxbai-embed-large"
MODEL_CHAT = "deepseek-r1"

def resetar_chat() -> None:
    st.session_state.messages = []


def extrair_resposta(texto):
    texto_sem_pensamento = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto_sem_pensamento.strip()


def carregar_documentos(arquivos):
    """Carrega e divide PDFs enviados pelo usuário."""
    documentos = []
    for arquivo in arquivos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(arquivo.read())
            loader = PyPDFLoader(tmp.name)
            documentos.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documentos)


def construir_rag(docs):
    """Cria vetor de embeddings e cadeia RAG a partir dos documentos."""
    embeddings = OllamaEmbeddings(model=MODEL_EMBEDDING)
    vectorstore = Chroma.from_documents(docs, embeddings)
    prompt = ChatPromptTemplate.from_template(
        """Use os documentos a seguir para responder a pergunta.
        
        documentos: {documents}
        
        Pergunta: {question}
        
        """
    )
    llm = ChatOllama(model=MODEL_CHAT, temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    retriever = vectorstore.as_retriever()
    return retriever, rag_chain


def executar_pergunta(pergunta: str, retriever, rag_chain) -> str:
    docs = retriever.invoke(pergunta)
    textos = "\n".join(doc.page_content for doc in docs)
    return rag_chain.invoke({"question": pergunta, "documents": textos})


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

    st.title("📄💬 Assistente RAG para PDFs com Ollama + LangChain")
    st.caption(
        "Carregue documentos PDF, processe-os e faça perguntas sobre seu conteúdo com modelos locais via Ollama. "
        "Powered by LangChain 🦜🔗"
    )
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        st.session_state.rag_chain = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📂 Documentos", divider="gray")
        arquivos = st.file_uploader(
            "Envie um ou mais PDFs:",
            type="pdf",
            accept_multiple_files=True,
        )
        if st.button("Processar PDFs") and arquivos:
            docs = carregar_documentos(arquivos)
            retriever, rag_chain = construir_rag(docs)
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.success("PDF(s) processado(s) com sucesso!")
        
        
        if st.button("Excluir base de conhecimento"):
            if "retriever" in st.session_state:
                del st.session_state["retriever"]
            
        st.button("Limpar Conversa", on_click=resetar_chat)

    with col2:
        st.header("💬 Chat com o Assistente", divider="gray")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        pergunta = st.chat_input("Digite sua pergunta aqui...")
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
                    resposta_final = extrair_resposta(resposta)
                st.markdown(resposta_final)
                st.session_state.messages.append({"role": "ai", "content": resposta_final})

    st.markdown("---")
    st.caption("Desenvolvido com ❤️ por Núcleo de Ciência de Dados • Unimed Blumenau")


if __name__ == "__main__":
    main()
