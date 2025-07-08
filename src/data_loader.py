from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PATH_DOCUMENTS = '../data/Anexo_II_DUT_2021_RN_465.2021_RN628.2025_RN629.2025.pdf'

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
