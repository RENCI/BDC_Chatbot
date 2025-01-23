
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

from langchain_ollama import OllamaEmbeddings
from .rag.chain import strip_thought

def set_emb_llm():

    load_dotenv(override=True)

    COMPLETION_URL = os.getenv("COMPLETION_URL")
    COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    DB_PATH = os.getenv("DB_PATH")


    if COMPLETION_URL and COMPLETION_MODEL:
        llm = ChatOpenAI(base_url=COMPLETION_URL, model=COMPLETION_MODEL, temperature=0)
        print("model: ", COMPLETION_MODEL, "base_url: ", COMPLETION_URL)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        print("model: gpt-3.5-turbo")


    if EMBEDDING_URL and EMBEDDING_MODEL:
        # emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, temperature=0)
        emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL)
        print("model: ", EMBEDDING_MODEL, "base_url: ", EMBEDDING_URL)
    else:
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        print("model: text-embedding-3-small")

    return emb, llm | strip_thought, DB_PATH