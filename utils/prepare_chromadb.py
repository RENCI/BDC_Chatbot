import json
import re
import pandas as pd

from typing import AsyncIterator, Iterator

from datetime import datetime, timedelta



from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma


import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores.utils import filter_complex_metadata


import shutil

import argparse
from tqdm import tqdm

import chromadb
import uuid

import pickle



from .chroma.utils import create_chroma_client

from .utils import set_emb_llm

emb, llm, DB_PATH = set_emb_llm()


shutil.rmtree(DB_PATH, ignore_errors=True) 

chroma_client = create_chroma_client(docs_path="data/", 
                                     db_path=DB_PATH, 
                                     emb=emb, llm=llm, 
                                     use_summary=True, 
                                     file_name_list = ['events.pkl', 
                                                       'latest_updates.pkl', 
                                                       'fellows.pkl', 
                                                       'pages.pkl'], 
                                     doc_type_list = ['event', 
                                                      'update', 
                                                      'fellow', 
                                                      'page'])







