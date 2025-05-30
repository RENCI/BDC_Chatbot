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
from langchain_core.documents import Document


from pydantic import BaseModel, Field, field_validator
from typing import Optional



from datetime import datetime, timedelta, date


from langchain_core.output_parsers import StrOutputParser






from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

config = RailsConfig.from_path("config")
guardrails = RunnableRails(config, verbose=True, output_key="answer")




from typing import List




from typing import Any, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever, VectorStore




class RetrieverWithScore(VectorStoreRetriever):
    # init with vectorstore
    def __init__(self, vectorstore: VectorStore, **kwargs: Any) -> None:
        """Initialize with vectorstore."""
        super().__init__(vectorstore=vectorstore, **kwargs)
        print(self.__dict__)
    
    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        """Get docs, adding score information."""
        docs, scores = zip(
            *self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
        )
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None, **kwargs: Any
    ) -> List[Document]:

        return self._get_docs_with_query(query, kwargs)





summary_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following text in 1-3 sentences, return the summary ONLY, This is NOT a conversation. \\n\\n{text}")]
)
def get_summary(text: str, llm, min_text=300):
    if min_text is None:
        min_text = 0
    if len(text) < min_text:
        return text
    
    return (summary_prompt|llm).invoke({"text": text}).model_dump()['content']





def rag_chain_constructor(retriever, llm, vectorstore: VectorStore = None, retriever_top_k=5, score_threshold=0.5):
    if vectorstore is not None:
        print("using RetrieverWithScore")
        # retriever = RetrieverWithScore(vectorstore, search_type="similarity_score_threshold", search_kwargs={'score_threshold': score_threshold,'k':retriever_top_k})
        retriever = RetrieverWithScore(vectorstore, search_kwargs={'k':retriever_top_k})
    
    
    
    
    
    # TODO: improve this prompt (questions are not returned as is)
    contextualize_q_system_prompt = """You are an assistant, called "BioData Catalyst(BDC) Assistant", for question-answering tasks related to BioData Catalyst. \
Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Replace all "BDC" in user input with "BioData Catalyst". \
If the question can be understood WITHOUT the chat history, do NOT change the question, return it as is. \
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
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


# Use 1 paragraph and keep the answer concise, unless otherwise specified.\
    qa_system_prompt = """You are an assistant, called "BioData Catalyst(BDC) Assistant", for question-answering tasks related to BioData Catalyst. (BDC only stands for BioData Catalyst, not other organizations)\
Use the following pieces of retrieved context to answer the question. \
If you can't get an answer base on the context, just say that you don't know. \
Keep the answer concise, prioritize using 1 paragraph, and include the most relevant information, unless a lengthier answer is required to answer the question or otherwise specified. \
You can use bullet points and markdown formatting if either is needed.\
The context are retrieved based on the user query and the chat history.\
If there is context provided, answer the question based on the context.\
Use the term 'documentation' instead of context in your repsponses.\

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
    
    return guardrails | rag_chain


class date_filter_params(BaseModel):
    start_date: Optional[int] = None
    end_date: Optional[int] = None


    @field_validator("start_date", mode="before")
    def validate_start_date(cls, v):
        if not v is None:
            if isinstance(v, datetime):
                v = v.timestamp()
            if isinstance(v, date):
                v = datetime.combine(v, datetime.min.time()).timestamp()
            if not isinstance(v, int):
                v = int(v)
        
        return v
        
    @field_validator("end_date", mode="before")
    def validate_end_date(cls, v):
        if not v is None:
            if isinstance(v, datetime):
                v = v.timestamp()
            if isinstance(v, date):
                v = datetime.combine(v, datetime.max.time()).timestamp()
            if not isinstance(v, int):
                v = int(v)
        
        return v
        

def construct_time_filter(search_query: date_filter_params = None):
    
    if search_query is None:
        search_query = date_filter_params(start_date=(datetime.now() - timedelta(days=7)).timestamp())
    
    
    
    # only filter by timestamp if timestamp exists in attribute
    print(search_query)
    
    comparisons = []
    if search_query.start_date is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.GTE,
                attribute="timestamp",
                value=search_query.start_date,
            )
        )
    if search_query.end_date is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.LTE,
                attribute="timestamp",
                value=search_query.end_date,
            )
        )
    
    # OR doc_type in in ['event', 'news']
    timestamp_DNE = Comparison(
        comparator=Comparator.NIN,
            attribute="doc_type",
            value=['event',],
    )
    
    
    
    if len(comparisons) == 0:
        return None
    elif len(comparisons) == 1:
        rag_filter = ChromaTranslator().visit_operation(
            Operation(operator=Operator.OR, 
                      arguments=[timestamp_DNE, comparisons[0]]))
    
    else:
        rag_filter = ChromaTranslator().visit_operation(
            Operation(operator=Operator.OR, 
                      arguments=[timestamp_DNE, Operation(operator=Operator.AND, arguments=comparisons)])
        )
    
    print('rag_filter: ', rag_filter)
    
    return rag_filter









def create_chunk_contextualizer_chain(llm, use_metadata_context=False):
    
    if use_metadata_context:
        contextualize_prompt = ChatPromptTemplate.from_template(
"""<metadata_context>
{context}
</metadata_context>
Here is the chunk we want to situate with the metadata context
<chunk>
{chunk_content}
</chunk>

Please give a short succinct natural language context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
""")
    
    else:
        contextualize_prompt = ChatPromptTemplate.from_template(
"""<document>
{context}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
""")

    contextualize_prompt += """

"""


    # Chain components
    chain = (
        contextualize_prompt 
        | llm 
        | StrOutputParser()
    )

    return chain






