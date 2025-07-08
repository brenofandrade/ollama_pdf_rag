import streamlit as st
import logging
import ollama
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings('ignore', category=UserWarning)

# Configuração da página
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

logger = logging.getLogger(__name__)


def extract_model_names(models_info):
    """
    Extrai os nomes dos modelos disponíveis no Ollama.
    """
    logger.info("Extraindo nomes dos modelos do Ollama")
    try:
        if hasattr(models_info, "models"):
            return tuple(model.model for model in models_info.models)
        return tuple()
    except Exception as e:
        logger.error(f"Erro ao extrair modelos: {e}")
        return tuple()


def main():
    # Título e descrição
    st.title("📄💬 Assistente RAG para PDFs com Ollama + LangChain")
    st.caption(
        "Carregue documentos PDF, processe-os e faça perguntas sobre seu conteúdo com modelos locais via Ollama. "
        "Powered by LangChain 🦜🔗"
    )
    st.markdown("---")

    # Layout principal
    col1, col2 = st.columns([1, 2])

    #### Configuração do Modelo
    with col1:
        st.header("⚙️ Configuração", divider="gray")
        models_info = ollama.list()
        available_models = extract_model_names(models_info)

        if available_models:
            selected_model = st.selectbox(
                "Selecione o modelo disponível:",
                available_models,
                key="model_select"
            )
            st.success(f"Modelo selecionado: **{selected_model}**")
        else:
            st.error("Nenhum modelo Ollama encontrado no sistema.")

        # use_sample = st.toggle(
        #     "Usar PDF de exemplo",
        #     help="Usa um documento pré-carregado para demonstração."
        # )

    #### Upload de Documentos
    with col1:
        st.header("📂 Documentos", divider="gray")
        uploaded_files = st.file_uploader(
            "Envie um ou mais PDFs:",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_upload"
        )
        if uploaded_files:
            st.info(f"{len(uploaded_files)} arquivo(s) carregado(s) com sucesso.")

    #### Área do Chat
    with col2:
        st.header("💬 Chat com o Assistente", divider="gray")

        chat_placeholder = st.empty()
        input_placeholder = st.chat_input("Digite sua pergunta aqui...")

        if input_placeholder:
            with chat_placeholder.chat_message("user"):
                st.write(input_placeholder)

            # Placeholder para resposta do assistente
            with chat_placeholder.chat_message("ai"):
                st.write("Estou processando sua pergunta...")

        else:
            st.info("Envie uma pergunta para começar a conversar com o assistente.")

    st.markdown("---")
    st.caption(
        """
        Desenvolvido com ❤️ por Núcleo de Ciência de Dados • Unimed Blumenau
        """
        )


if __name__ == "__main__":
    main()
