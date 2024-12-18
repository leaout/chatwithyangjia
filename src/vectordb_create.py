#vector-db-create.py
# create a vector database from a pdf file
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

current_directory = os.getcwd()
relative_path = os.path.join(current_directory, './data/yangjia.txt')

print(relative_path)    
loaders = [TextLoader(relative_path, encoding='utf-8')]

docs = []
for file in loaders:
    docs.extend(file.load())
#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#print(len(docs))

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_test1")

print(vectorstore._collection.count())