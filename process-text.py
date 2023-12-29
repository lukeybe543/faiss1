from langchain.llms import HuggingFaceHub
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from zipfile import ZipFile
import os
import shutil
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

load_dotenv()
apikey = os.getenv("OPENAI_API_KEY")

text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
foo = Document(page_content='foo is fou!',metadata={"source":'foo source'})

def is_duplicate(split_docs, db):
    epsilon = 0.0
    print(f"DUPLICATE: Treating: {split_docs[0].metadata['source'].split('/')[-1]}")
    for doc in split_docs[:min(3,len(split_docs))]:
        _, score = db.similarity_search_with_score(doc.page_content, k=1)[0]
        epsilon += score
    print(f"DUPLICATE: epsilon: {epsilon}")
    return epsilon < 0.05

#main function that iterates over all pdfs in script folder
def main(directory):
    database_name = input("Please enter the database name: ")
    if len(database_name)==0:
        database_name = "faissdb"
    try:
        if os.path.isfile(f"{database_name}/index.faiss"):
            print(f"Loading local database: {database_name}")
            db = FAISS.load_local(database_name,embeddings)
        else:    
            print(f"SESSION: {database_name} database does not exist, create a FAISS db")
            db =  FAISS.from_documents([foo], embeddings)
            db.save_local(database_name)
            print(f"SESSION: {database_name} database created")
    except FileNotFoundError as e: 
        print(f"Failed to load or create database due to error: {str(e)}")
    document_path = os.path.join(os.getcwd(),'transcripts-youtube')
    loader = DirectoryLoader(path=document_path,show_progress=True)
    print("loading docs")
    docs = loader.load()
    print("embedding docs")
    split_text = text_splitter.split_documents(docs)
    print(f"embedding into vector database")
    db1 = FAISS.from_documents(split_text,embeddings)
    print(f"merging embeddings into {database_name}")
    db.merge_from(db1)
    print(f"saving {database_name}")
    db.save_local(database_name)
    print(f"saving {database_name}")
  
if __name__ == "__main__":
    directory = r"C:\Users\Luke Bociulis\Syncthing\share-folder\pdf-db\transcripts-youtube"
    main(directory)