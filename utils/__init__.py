
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langserve import RemoteRunnable

from .rag.chain import strip_thought

def set_emb_llm():

    load_dotenv(override=True)

    COMPLETION_URL = os.getenv("COMPLETION_URL")
    COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    GUARDIAN_MODEL = os.getenv("GUARDIAN_MODEL")
    GUARDIAN_URL = os.getenv("GUARDIAN_URL")
    DUGBOT_RUNNABLE_URL = os.getenv("DUGBOT_RUNNABLE_URL")
    DB_PATH = os.getenv("DB_PATH")


    if COMPLETION_URL and COMPLETION_MODEL:
        llm = ChatOpenAI(base_url=COMPLETION_URL, model=COMPLETION_MODEL, temperature=0)
        print("model: ", COMPLETION_MODEL, "base_url: ", COMPLETION_URL)
    else:
        COMPLETION_MODEL = "gpt-3.5-turbo" if COMPLETION_MODEL is None else COMPLETION_MODEL
        llm = ChatOpenAI(model=COMPLETION_MODEL, temperature=0)
        print("model: ", COMPLETION_MODEL)


    if EMBEDDING_URL and EMBEDDING_MODEL:
        # emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, temperature=0)
        emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL)
        print("model: ", EMBEDDING_MODEL, "base_url: ", EMBEDDING_URL)
    else:
        EMBEDDING_MODEL = "text-embedding-3-small" if EMBEDDING_MODEL is None else EMBEDDING_MODEL
        emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        print("model: ", EMBEDDING_MODEL)

    if GUARDIAN_MODEL and GUARDIAN_URL:
        guardian_llm = ChatOllama(base_url=GUARDIAN_URL, model=GUARDIAN_MODEL)
        print("model: ", GUARDIAN_MODEL, "base_url: ", GUARDIAN_URL)
    else:
        guardian_llm = llm # default to llm
    
    if DUGBOT_RUNNABLE_URL:
        dugbot_chain = RemoteRunnable(url=DUGBOT_RUNNABLE_URL)
    else:
        dugbot_chain = None
    
    return emb, llm | strip_thought, guardian_llm, dugbot_chain, DB_PATH

