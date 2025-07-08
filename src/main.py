import streamlit as st
import logging
import ollama
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from data_embedding import vectorstore

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

# Prompt template used for RAG interactions
prompt = ChatPromptTemplate.from_template(
    "Use os documentos a seguir para responder a pergunta.\n{documents}\nPergunta: {question}"
)

# Initialize the LLM with Llama 3.2 model
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Retriever from the embedded documents
retriever = vectorstore.as_retriever()

def resetar_chat():
    st.session_state = []
    st.session_state.messages = []


class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question: str) -> str:
        """Run the RAG pipeline for a given question."""
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join(doc.page_content for doc in documents)
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


rag_application = RAGApplication(retriever, rag_chain)

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

    with col1:
        st.header("⚙️ Configuração", divider="gray")

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
        st.button(on_click=resetar_chat, label="Limpar")
    with col2:
        st.header("💬 Chat com o Assistente", divider="gray")

        # Histórico de mensagens na sessão
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Exibir histórico
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Caixa de input para nova pergunta
        user_input = st.chat_input("Digite sua pergunta aqui...")

        if user_input:
            # Mostra pergunta do usuário
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Chama o pipeline RAG para resposta
            with st.chat_message("ai"):
                st.write("Estou processando sua pergunta...")
                try:
                    resposta = rag_application.run(user_input)
                except Exception as e:
                    resposta = f"Erro ao processar pergunta: {e}"
                st.write(resposta)
                st.session_state.messages.append({"role": "ai", "content": resposta})

    st.markdown("---")
    st.caption("""Desenvolvido com ❤️ por Núcleo de Ciência de Dados • Unimed Blumenau""")

if __name__ == "__main__":
    main()
