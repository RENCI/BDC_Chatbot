from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

from langchain.retrievers.document_compressors import FlashrankRerank
from utils.rag.chain import create_main_chain, create_time_filter
from langchain_chroma import Chroma
from utils import set_emb_llm
from langchain.globals import set_debug, set_verbose

set_debug(True)

def init_vars(retriever_top_k = 5, default_rag_filter = None, rerank_top_k = 5):
    emb, llm, guardian_llm, DB_PATH = set_emb_llm()

    vectorstore = Chroma(persist_directory=DB_PATH,embedding_function=emb)

    if default_rag_filter is None:
        default_rag_filter = create_time_filter()
    
    default_retriever = vectorstore.as_retriever(
        search_kwargs = {"k": retriever_top_k, 
                         "filter": default_rag_filter, 
                         })
    default_retriever.search_type = "similarity_score_threshold"
    
    if rerank_top_k > 0:
        compressor = FlashrankRerank(top_n=rerank_top_k, model = "ms-marco-MiniLM-L-12-v2")
    elif rerank_top_k == 0: 
        compressor = FlashrankRerank(top_n=retriever_top_k, model = "ms-marco-MiniLM-L-12-v2")
    else:
        compressor = None
    
    
    return llm, guardian_llm, emb, vectorstore, default_retriever, retriever_top_k, compressor

llm, guardian_llm, emb, vectorstore, default_retriever, retriever_top_k, compressor = init_vars(retriever_top_k=20, 
                                                                                  rerank_top_k=10)

default_rag_chain = create_main_chain(default_retriever, llm, guardian_llm, emb, vectorstore, retriever_top_k=retriever_top_k, score_threshold=0.5, compressor=compressor, hybrid_retriever=True)




app = FastAPI(
    title="BDC Bot",
    version="1.0",
    description="BDC Bot",
)




# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)



@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, 
           default_rag_chain,
           path="/bdc-bot")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
