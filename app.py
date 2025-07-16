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
from feedback import salvar_feedback

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 0
MODEL_EMBEDDING = "mxbai-embed-large"
MODEL_CHAT = "llama3.2:latest"

def resetar_chat() -> None:
    st.session_state.messages = []


# def extrair_resposta(texto):
#     return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()


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
        """
        Você é um assistente especializado em responder perguntas com base em documentos internos da empresa.  

        Sempre que possível, organize a resposta em tópicos claros, objetivos e relevantes.  
        Se a informação não estiver presente nos documentos, apenas responda: **"Não há informações disponíveis nos documentos fornecidos para responder a esta pergunta."**  

        Use apenas as informações presentes nos documentos abaixo e não invente ou assuma nada além do que está explicitamente dito.  
        Se os documentos contiverem mais de uma informação relevante, organize em uma lista numerada.  

        ### Documentos:
        {documents}

        ### Pergunta:
        {question}

        ### Resposta:
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

        avaliacao_checkbox = st.checkbox("Feedback", value=False)

    
    with st.container():
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
                    
                st.markdown(resposta)
                st.session_state.messages.append({"role": "ai", "content": resposta})
    
    with st.container():
        st.empty()
    
    col_1, col_2 = st.columns([2, 2])
    
    
    with col_1:
        st.empty()
    with col_2:
        if avaliacao_checkbox:
            with st.container():
                st.header("✍️ Feedback", divider="gray")
                avaliacao = str(st.feedback("thumbs"))
                usuario = st.text_input("Seu nome", "")
                mensagem = st.text_area("Sua mensagem", "")

                if st.button("Enviar"):
                    if all([avaliacao, usuario, mensagem]):
                        salvar_feedback(usuario=usuario, mensagem=mensagem, avaliacao=avaliacao)
                        st.success("✅ Feedback salvo com sucesso!")
                    else:
                        st.warning("⚠️ Preencha todos os campos antes de enviar.")
        else:
            st.empty()



    st.markdown("---")
    st.caption("Desenvolvido com ❤️ por Núcleo de Ciência de Dados • Unimed Blumenau")


if __name__ == "__main__":
    main()
