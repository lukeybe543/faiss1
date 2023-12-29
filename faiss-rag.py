from email.headerregistry import UnstructuredHeader
import dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import DirectoryLoader
import calibre_search as cali



loader = DirectoryLoader('./documents/', glob="./*.pdf", loader_cls=UnstructuredPDFLoader)
