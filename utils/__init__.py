
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langserve import RemoteRunnable

from .rag.chain import strip_thought

def set_emb_llm():

    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    
    COMPLETION_MODEL_PROVIDER = os.getenv("COMPLETION_MODEL_PROVIDER")
    EMBEDDING_MODEL_PROVIDER = os.getenv("EMBEDDING_MODEL_PROVIDER")
    GUARDIAN_MODEL_PROVIDER = os.getenv("GUARDIAN_MODEL_PROVIDER")

    
    COMPLETION_URL = os.getenv("COMPLETION_URL")
    COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    GUARDIAN_MODEL = os.getenv("GUARDIAN_MODEL")
    GUARDIAN_URL = os.getenv("GUARDIAN_URL")
    
    
    
    DUGBOT_RUNNABLE_URL = os.getenv("DUGBOT_RUNNABLE_URL")
    DB_PATH = os.getenv("DB_PATH")

    
    
    # completion model
    print("COMPLETION_MODEL")
    if COMPLETION_MODEL_PROVIDER == "openai":
        if not COMPLETION_MODEL:
            COMPLETION_MODEL = "gpt-4o-mini"
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=COMPLETION_MODEL, temperature=0)
        print("provider: ", COMPLETION_MODEL_PROVIDER, "model: ", COMPLETION_MODEL)
    
    elif COMPLETION_MODEL_PROVIDER == "google":
        if not COMPLETION_MODEL:
            COMPLETION_MODEL = "gemini-2.0-flash"
        llm = ChatGoogleGenerativeAI(api_key=GOOGLE_AI_API_KEY, model=COMPLETION_MODEL, temperature=0)
        print("provider: ", COMPLETION_MODEL_PROVIDER, "model: ", COMPLETION_MODEL)
    
    elif COMPLETION_MODEL_PROVIDER == "vllm":
        if not COMPLETION_MODEL or not COMPLETION_URL:
            raise ValueError("COMPLETION_MODEL and COMPLETION_URL must be set")
        llm = ChatOpenAI(base_url=COMPLETION_URL, model=COMPLETION_MODEL, temperature=0)
        print("provider: ", COMPLETION_MODEL_PROVIDER, "model: ", COMPLETION_MODEL, "base_url: ", COMPLETION_URL)
    
    elif COMPLETION_MODEL_PROVIDER == "ollama":
        if not COMPLETION_MODEL or not COMPLETION_URL:
            raise ValueError("COMPLETION_MODEL and COMPLETION_URL must be set")
        llm = ChatOllama(base_url=COMPLETION_URL, model=COMPLETION_MODEL, temperature=0)
        print("provider: ", COMPLETION_MODEL_PROVIDER, "model: ", COMPLETION_MODEL, "base_url: ", COMPLETION_URL)
    
    else:
        raise ValueError("Invalid COMPLETION_MODEL_PROVIDER")
    
    
    
    
    
    # embedding model
    print("EMBEDDING_MODEL")
    if EMBEDDING_MODEL_PROVIDER == "openai":
        if not EMBEDDING_MODEL:
            EMBEDDING_MODEL = "text-embedding-3-small"
        emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        print("provider: ", EMBEDDING_MODEL_PROVIDER, "model: ", EMBEDDING_MODEL)
    
    elif EMBEDDING_MODEL_PROVIDER == "google":
        if not EMBEDDING_MODEL:
            EMBEDDING_MODEL = "models/text-embedding-004"
        emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        print("provider: ", EMBEDDING_MODEL_PROVIDER, "model: ", EMBEDDING_MODEL)
    
    elif EMBEDDING_MODEL_PROVIDER == "vllm":
        if not EMBEDDING_MODEL or not EMBEDDING_URL:
            raise ValueError("EMBEDDING_MODEL and EMBEDDING_URL must be set")
        emb = OpenAIEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, temperature=0)
        print("provider: ", EMBEDDING_MODEL_PROVIDER, "model: ", EMBEDDING_MODEL, "base_url: ", EMBEDDING_URL)
    
    elif EMBEDDING_MODEL_PROVIDER == "ollama":
        if not EMBEDDING_MODEL or not EMBEDDING_URL:
            raise ValueError("EMBEDDING_MODEL and EMBEDDING_URL must be set")
        emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL)
        print("provider: ", EMBEDDING_MODEL_PROVIDER, "model: ", EMBEDDING_MODEL, "base_url: ", EMBEDDING_URL)
    
    else:
        raise ValueError("Invalid EMBEDDING_MODEL_PROVIDER")
    
    
    
    # guardian model
    print("GUARDIAN_MODEL")
    if GUARDIAN_MODEL_PROVIDER == "openai":
        if not GUARDIAN_MODEL:
            GUARDIAN_MODEL = "gpt-4o-mini"
        guardian_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=GUARDIAN_MODEL, temperature=0)
        print("provider: ", GUARDIAN_MODEL_PROVIDER, "model: ", GUARDIAN_MODEL)
    elif GUARDIAN_MODEL_PROVIDER == "google":
        if not GUARDIAN_MODEL:
            GUARDIAN_MODEL = "gemini-2.0-flash"
        guardian_llm = ChatGoogleGenerativeAI(api_key=GOOGLE_AI_API_KEY, model=GUARDIAN_MODEL, temperature=0)
        print("provider: ", GUARDIAN_MODEL_PROVIDER, "model: ", GUARDIAN_MODEL)
    
    elif GUARDIAN_MODEL_PROVIDER == "vllm":
        if not GUARDIAN_MODEL or not GUARDIAN_URL:
            raise ValueError("GUARDIAN_MODEL and GUARDIAN_URL must be set")
        guardian_llm = ChatOpenAI(base_url=GUARDIAN_URL, model=GUARDIAN_MODEL, temperature=0)
        print("provider: ", GUARDIAN_MODEL_PROVIDER, "model: ", GUARDIAN_MODEL, "base_url: ", GUARDIAN_URL)
    
    elif GUARDIAN_MODEL_PROVIDER == "ollama":
        if not GUARDIAN_MODEL or not GUARDIAN_URL:
            raise ValueError("GUARDIAN_MODEL and GUARDIAN_URL must be set")
        guardian_llm = ChatOllama(base_url=GUARDIAN_URL, model=GUARDIAN_MODEL, temperature=0)
        print("provider: ", GUARDIAN_MODEL_PROVIDER, "model: ", GUARDIAN_MODEL, "base_url: ", GUARDIAN_URL)
    
    elif GUARDIAN_MODEL_PROVIDER == "default":
        guardian_llm = llm
        print("provider: ", COMPLETION_MODEL_PROVIDER, "model: ", COMPLETION_MODEL, "base_url: ", COMPLETION_URL)
    
    else:
        raise ValueError("Invalid GUARDIAN_MODEL_PROVIDER")
    
    

    
    if DUGBOT_RUNNABLE_URL:
        dugbot_chain = RemoteRunnable(url=DUGBOT_RUNNABLE_URL)
    else:
        dugbot_chain = None
    
    # return emb, llm | strip_thought, guardian_llm, dugbot_chain, DB_PATH
    return emb, llm, guardian_llm, dugbot_chain, DB_PATH

