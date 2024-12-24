import os
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

arquivo = "NT.018 - Rede de Distribuicao Compacta.pdf"
if not os.path.exists(arquivo):
    raise FileNotFoundError(f"O arquivo {arquivo} não foi encontrado!")
loader = PyPDFLoader(arquivo)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents=documents)


persist_directory = 'db'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

collection_name = 'norma_distribuicao'

embedding = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name=collection_name,
)


print("Coleção persistida com sucesso!")
print("Coleção persistida com sucesso!")
print(f"Persistência está sendo feita em: {persist_directory}")






