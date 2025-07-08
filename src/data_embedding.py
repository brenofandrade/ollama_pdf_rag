# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


PATH_DOCUMENTS = 'data/avaliacao_RAG.pdf'
DB_DIR = 'db/chromadb/'


chunk_size = 100
chunk_overlap = 50

loader = PyPDFLoader(PATH_DOCUMENTS)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    # separator=r"\n|\s+",
    # is_separator_regex=True
)

docs = text_splitter.split_documents(documents=documents)

MODEL_EMBEDDING = "mxbai-embed-large"

embeddings = OllamaEmbeddings(model=MODEL_EMBEDDING)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=DB_DIR,
    collection_name='documents'
)

print("Base vetorial criada com sucesso e persistida em:", DB_DIR)
