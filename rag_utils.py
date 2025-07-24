import shutil
import tempfile
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
MODEL_EMBEDDING = "mxbai-embed-large:latest"
MODEL_CHAT = "llama3.2:latest"

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
    vectorstore = Chroma.from_documents(
        docs, 
        embeddings, 
        # persist_directory='db'  # Persistir diretório pode provocar erro em inserir novos documentos.
        )

    ### Possiveis escolhas de prompt já testados ###
    # -------------------------------------------- #

    # prompt_path = f"prompt/prompt_template_v1.txt"
    # prompt_path = f"prompt/prompt_template_v2.txt"
    # prompt_path = f"prompt/prompt_template_v3.txt"
    # prompt_path = f"prompt/prompt_template_v4.txt"
    prompt_path = f"prompt/prompt_template_v5.txt"

    with open(prompt_path, "r", encoding='utf-8') as file:
        prompt_file = file.read()

    print(f"TEMPLATE: \n\n{prompt_file}")

    prompt = ChatPromptTemplate.from_template(prompt_file)

    llm = ChatOllama(model=MODEL_CHAT, temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    retriever = vectorstore.as_retriever()
    return retriever, rag_chain


def executar_pergunta(pergunta: str, retriever, rag_chain) -> str:
    docs = retriever.invoke(pergunta)
    textos = "\n".join(doc.page_content for doc in docs)
    return rag_chain.invoke({"question": pergunta, "documents": textos})


def deletar_base_conhecimento(path:str = "db"):
    """
    Deleta a base de conhecimento
    """

    try:
        shutil.rmtree(path)
        logging.info(f"Base de conhecimento removida de :{path}")
    except FileNotFoundError:
        logging.warning(f"Base de conhecimento não encontrada em: {path}")
    except Exception as error:
        logging.error(f"Erro ao tentar remover base de conhecimento: {error}")

