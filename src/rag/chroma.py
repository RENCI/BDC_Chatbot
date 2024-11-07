# import json
# import re
# import pandas as pd

# from typing import AsyncIterator, Iterator

# from datetime import datetime, timedelta



# from langchain_core.document_loaders import BaseLoader
# from langchain_core.documents import Document

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_chroma import Chroma


import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings

# from langchain_community.vectorstores.utils import filter_complex_metadata


import shutil

# import argparse
# from tqdm import tqdm

import chromadb
import uuid

import pickle




load_dotenv(override=True)

EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
DB_PATH = os.getenv("DB_PATH")

shutil.rmtree(DB_PATH, ignore_errors=True)


if EMBEDDING_URL and EMBEDDING_MODEL:
    emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, show_progress=True)
    print("model: ", EMBEDDING_MODEL, "base_url: ", EMBEDDING_URL)
else:
    emb = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
    print("model: text-embedding-3-small")



def loadPKL(file_path, doc_type):
    with open(file_path, 'rb') as f:
        pkl_list = pickle.load(f)
    page_contents = [row['content'] for row in pkl_list]
    metadata = [row['metadata'] for row in pkl_list]
    
    embeddings = emb.embed_documents(page_contents)
    return page_contents, metadata, embeddings
    

def create_chroma_client(docs_path, db_path = ".chroma_db/", emb = emb):
    file_name_list = ['events.pkl', 'latest_updates.pkl', 'fellows.pkl', 'pages.pkl']
    doc_type_list = ['event', 'update', 'fellow', 'page']
    
    all_contents = []
    all_metadatas = []
    all_embeddings = []
    
    for file_name, doc_type in zip(file_name_list, doc_type_list):
        print("Loading ", doc_type)
        page_contents, metadata, embeddings = loadPKL(os.path.join(docs_path, file_name), doc_type)
        all_contents.extend(page_contents)
        all_metadatas.extend(metadata)
        all_embeddings.extend(embeddings)
    
    for metadata in all_metadatas:
        if len(metadata) == 0:
            metadata = None
    
    all_ids = [str(uuid.uuid4()) for _ in range(len(all_contents))]
    
    persistent_client = chromadb.PersistentClient(path=db_path)
    collection = persistent_client.get_or_create_collection("langchain")
    collection.add(ids=all_ids, documents=all_contents, embeddings=all_embeddings, metadatas=all_metadatas)
    
    
    return persistent_client





















