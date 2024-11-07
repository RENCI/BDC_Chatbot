from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# from langchain.chat_message_histories import ChatMessageHistory

from langchain.chains import create_retrieval_chain#, create_stuff_documents_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever


from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
# from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_community.query_constructors.chroma import ChromaTranslator

from datetime import datetime, timedelta



def rag_chain_constructor(retriever, llm):
    

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # history_aware_retriever = create_history_aware_retriever(
    #     ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"), retriever, contextualize_q_prompt
    # )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )



    qa_system_prompt = """You are an assistant, called "BioData Catalyst(BDC) Assistant", for question-answering tasks related BioData Catalyst. \
Use the following pieces of retrieved context to answer the question. \
If you can't get an answer base on the context, just say that you don't know. \
Use 1-3 sentences and keep the answer concise, unless otherwise specified.\
The context are retrieved based on the user query and the chat history.\
If there is context provided, answer the question based on the context.\

### context: {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # question_answer_chain = create_stuff_documents_chain(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"), qa_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


class Search(BaseModel):
    start_date: Optional[int]
    end_date: Optional[int]


def construct_time_filter():
    search_query = Search(start_date=(datetime.now() - timedelta(days=7)).timestamp())
    
    
    
    comparisons = []
    if query.start_date is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.GTE,
                attribute="timestamp",
                value=query.start_date,
            )
        )
    














