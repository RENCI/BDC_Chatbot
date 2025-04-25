from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

from langchain.retrievers.document_compressors import FlashrankRerank
from utils.rag.chain import create_main_chain, create_time_filter, create_router_chain, create_query_classifier_chain, create_input_guardrail_chain
from langchain_chroma import Chroma
from utils import set_emb_llm
from langchain.globals import set_debug, set_verbose

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from langchain.schema.runnable import RunnableBranch, RunnablePassthrough

set_debug(True)

def init_vars(retriever_top_k = 5, default_rag_filter = None, rerank_top_k = 5):
    emb, llm, guardian_llm, dugbot_chain, DB_PATH = set_emb_llm()

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
    
    
    return llm, guardian_llm, dugbot_chain, emb, vectorstore, default_retriever, retriever_top_k, compressor

llm, guardian_llm, dugbot_chain, emb, vectorstore, default_retriever, retriever_top_k, compressor = init_vars(retriever_top_k=20, 
                                                                                  rerank_top_k=10)

bdcbot_chain = create_main_chain(default_retriever, llm, guardian_llm, emb, vectorstore, retriever_top_k=retriever_top_k, score_threshold=0.5, compressor=compressor, hybrid_retriever=True)

print("bdcbot_chain:", bdcbot_chain)
print("dugbot_chain:", dugbot_chain)

if dugbot_chain:
    classifier_chain = create_query_classifier_chain(llm)
    main_chain = create_router_chain(bdcbot_chain, dugbot_chain, classifier_chain, llm)
else:
    main_chain = bdcbot_chain    

# region: add guardrails
guardrails_config = RailsConfig.from_path("config")
guardrails = RunnableRails(guardrails_config, 
                            llm=guardian_llm,
                            verbose=True, 
                            output_key="answer")

input_guardrail_chain = create_input_guardrail_chain(llm)





main_chain_branch = RunnableBranch(
    (lambda x: not x["blocked"], main_chain),
    # return guardrail result
    lambda x: x
) 

chatbot_chain = input_guardrail_chain | main_chain_branch

# endregion
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
        #   chatbot_chain,
          guardrails | main_chain,
          path="/bdc-bot")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
