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

load_dotenv()
apikey = os.getenv("OPENAI_API_KEY")

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
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

def merge_split_docs_to_db(split_docs, db):
    if not split_docs:
        print("MERGE to db: NO docs!!")
        return db

    filename = split_docs[0].metadata['source']
    if is_duplicate(split_docs,db):
        print(f"MERGE: Document is duplicated: {filename}")
        return db
    print(f"MERGE: number of split docs: {len(split_docs)}")
    batch = 1000
    for i in range(0, len(split_docs), batch):
        print(f"Merging chunks {i} to {i+batch}...")
        db1 = FAISS.from_documents(split_docs[i:i+batch], embeddings)
        db.merge_from(db1)
        print(f"Finished merging split docs for file: {filename}")
    return db


def merge_pdf_to_db(filename,db):
  doc = PyPDFLoader(filename).load()
  doc[0].metadata['source'] = os.path.basename(filename)
  split_docs = text_splitter.split_documents(doc)
  return merge_split_docs_to_db(split_docs,db)

#main function that iterates over all pdfs in script folder
def main(directory):
    database_name = input("Please enter the database name: ")
    split_docs=[]
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

    file_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    total_files = len(file_paths)
    print("EMBEDDED, before embedding: ", database_name, len(db.index_to_docstore_id))

    for idx, filename in enumerate(file_paths):
        source = filename.split('/')[-1]
        print(f"Processing file {idx + 1} of {total_files}: {source}")

        if filename.endswith('.pdf'):
            print(f"Merging pdf {filename} to DB:{database_name}")
            db = merge_pdf_to_db(filename, db)
            print(f"Saving DB:{database_name}")
            db.save_local(database_name)
            
if __name__ == "__main__":
    directory = r"C:\Users\Luke Bociulis\Syncthing\share-folder\pdf-db\History_Revisionist"
    main(directory)