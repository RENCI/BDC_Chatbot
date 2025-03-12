from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# from langchain.chat_message_histories import ChatMessageHistory

from langchain.chains import create_retrieval_chain#, create_stuff_documents_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage

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


from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Any, Dict, ClassVar, Set, List, Iterable, Optional


from datetime import datetime, timedelta, date





from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails






from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever, VectorStore


from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough


from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


import re

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

import yaml



def load_yaml(yaml_path: str):
    with open(yaml_path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def merge_responses(x):
    # append, display with added disclaimer
    if x["flag"] == 'a':
        # x["display_answer"] = f"{x['answer']}\n\n<disclaimer>{x['response']}</disclaimer>"
        x["display_answer"] = f"{x['answer']}\n\n{x['response']}"
        return x
    # replace, display replaced text
    elif x["flag"] == 'r':
        x["answer"] = f"Answer was replaced with predefined response: \n{x['response']}"
        # x["display_answer"] = f"<replaced>{x['response']}</replaced>"
        x["display_answer"] = f"{x['response']}"
        return x
    # default, display answer
    elif x["flag"] == 'd':
        x["display_answer"] = x["answer"]
        return x
    # error
    else:
        raise ValueError(f"Invalid flag: {x['flag']}")


def fix_data_type(x):
    # Document to dict
    # relevance_score to float
    if "context" in x:
        print("have context")
        for i, doc in enumerate(x["context"]):
            doc = doc.dict()
            if "metadata" in doc and "relevance_score" in doc["metadata"]:
                print("have relevance_score")
                doc["metadata"]["relevance_score"] = doc["metadata"]["relevance_score"].item()
            x["context"][i] = doc
    return x


def strip_thought(message: AIMessage):
    messages = message.content.split('</think>')
    thought = messages[0].replace('<think>', '').replace('</think>', '')
    message.content = messages[-1].strip("\n\n")
    message.response_metadata['thought'] = thought
    return message



class VectorStoreRetrieverWithScore(VectorStoreRetriever):

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
            doc.metadata["score"] = float(score)
            doc.metadata["retriever_type"] = "similarity"
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None, **kwargs: Any
    ) -> List[Document]:

        return self._get_docs_with_query(query, kwargs)





from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun










class BM25RetrieverWithScore(BM25Retriever):

    emb: Any = Field(default=None, exclude=True)
        
    def __init__(self, emb=None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.emb = emb
    


    
    @classmethod
    def from_documents(cls, emb, **kwargs: Any) -> "BM25RetrieverWithScore":
        retriever = super(BM25RetrieverWithScore, cls).from_documents(**kwargs)
        retriever.emb = emb
        return retriever
    
    # declear emb
    # emb = None
    
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        
        # print("return_docs[0]: ", return_docs[0])
        
        
        # get similarity score of query to each doc
        doc_emb = self.emb.embed_documents([doc.page_content for doc in return_docs])
        query_emb = self.emb.embed_query(query)
        
        for i, doc in enumerate(return_docs):
            doc.metadata["score"] = float(cosine_similarity([query_emb], [doc_emb[i]])[0][0])
            
            doc.metadata["retriever_type"] = "bm25"
        
        
        return return_docs




def get_summary(text: str, llm):
    summary_prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following text in 1-3 sentences, return the summary ONLY, This is NOT a conversation. \\n\\n{text}")]
    )
    return (summary_prompt|llm).invoke({"text": text}).model_dump()['content']

def create_topic_classifier_chain(topics: List[str], llm):
    """Creates a chain that classifies user queries into predefined topics."""
    
    # Create a custom model class with the allowed topics
    ModelWithTopics = type(
        'ModelWithTopics',
        (TopicClassification,),
        {'allowed_topics': set(topics)}
    )
    
    topic_list = ", ".join([f'"{topic}"' for topic in topics])
    
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a topic classifier. Given a user query, determine if it's related to any of the following topics: {topic_list}.
        If the query clearly relates to one of these topics, return ONLY that topic name from the list.
        If it doesn't clearly match any topic, return ONLY "other".
        Return ONLY the topic name from the list or "other" with no additional text or explanation.
        The response can ONLY be a topic name from the list or "other". """),
        ("human", "{input}")
    ])
    
    return (
        classifier_prompt 
        | llm 
        | StrOutputParser() 
        | (lambda x: {"topic": x}) 
        | (lambda x: ModelWithTopics(**x).topic) # return string
    )

class TopicClassification(BaseModel):
    topic: str
    allowed_topics: ClassVar[Set[str]]
    
    @model_validator(mode='after')
    def validate_topic(self):
        """Ensure topic is either in allowed_topics or 'other'"""
        if not hasattr(self, 'allowed_topics'):
            return self
        if self.topic != "other" and self.topic not in self.allowed_topics:
            raise ValueError(f"Topic must be one of {self.allowed_topics} or 'other'")
        return self


def create_predefined_response_chain(predefined_responses, llm):
    
    topics_list = list(predefined_responses.keys())

    classifier_chain = create_topic_classifier_chain(topics_list, llm)
    
    # region: predefined branch and fallback
    predefined_chain = RunnableLambda(
        lambda x: {
            "input": x["input"],
            "chat_history": x["chat_history"],
            "context": [],
            "topic": x["topic"],
            "response": predefined_responses[x["topic"]]["response"],
            "flag": predefined_responses[x["topic"]]["flag"],

        }
    )

    predefined_chain_fallback = RunnableLambda(
        lambda x: {
            "input": x["input"],
            "chat_history": x["chat_history"],
            "context": [],
            "topic": x["topic"],
            "response": None,
            "flag": 'd',

        }
    )
    # endregion: predefined branch and fallback
    
    topic_branch = RunnableBranch(
        (lambda x: x["topic"] in topics_list, predefined_chain),
        predefined_chain_fallback
    )


    return (
        RunnablePassthrough.assign(
            topic=classifier_chain
        ) | topic_branch
    )








def create_qa_rag_chain(retriever, llm):
    
    contextualize_q_system_prompt = """You are an assistant, called "BDC Bot", for question-answering tasks related to NHLBI BioData Catalyst®️. \
Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Replace "NHLBI BioData Catalyst®️", "BioData Catalyst", or any short form of it in user input with "BDC". \
Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
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
    qa_system_prompt = """You are an assistant, called "BDC Bot", for question-answering tasks related to NHLBI BioData Catalyst®️ (BDC). (BDC only stands for BioData Catalyst, not other organizations)\
Use the following pieces of retrieved context to answer the question. \
If you can't get an answer base on the context, just say that you don't know. \
Keep the answer concise, prioritize using 1 paragraph, and include the most relevant information, unless a lengthier answer is required to answer the question or otherwise specified. \
You can use bullet points and markdown formatting if either is needed.\
The context are retrieved based on the user query and the chat history.\
If there is context provided, answer the question based on the context.\
Use the term 'documentation' instead of context in your repsponses.\
DO NOT USE "NHLBI BioData Catalyst®️" or any short form of it. You MUST ONLY refer it as ```BDC``` in your responses, even if the user query is not refering it as BDC.\

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



def create_bdc_response_regex_chain():
    def process_bdc_names(x):
        
        def remove(match):
            pre = " " if match.group('pre') else ""
            post = " " if match.group('post') else ""
            return " " if pre and post else pre + post

        def replace(match):
            pre = " " if match.group('pre') else ""
            post = " " if match.group('post') else ""
            return (" BDC " if pre and post else pre + "BDC" + post)

        
        if "answer" in x and x["answer"]:
            text = x["answer"]
            # remove terms in parentheses
            text = re.sub(
                r'(?P<pre>\s*)\(\s*(?:(?:NHLBI\s+)?BioData\s+Catalyst(?:®️)?|BDC)\s*\)(?P<post>\s*)',
                remove,
                text
            )
            
            # replace terms with "BDC"
            text = re.sub(
                r'(?P<pre>\s*)(?:NHLBI\s+)?BioData\s+Catalyst(?:®️)?(?P<post>\s*)',
                replace,
                text
            )

            x["answer"] = text
            
        return x
    
    return RunnableLambda(process_bdc_names)


def create_bdc_response_llm_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rewrite the following text, replacing all instances of "NHLBI BioData Catalyst®️", "BioData Catalyst", and any short form of it with the abbreviation "BDC". If a paratheses surrounded "(BDC)" after the replacement, remove it. If there is clear indication that the text is only explaining what the abbreviation stands for, keep it as is.
        Keep all other content exactly the same, including formatting, punctuation, and line breaks.
        Return ONLY the rewritten text without any additional explanation or commentary."""),
        ("human", "{text}")
    ])
    
    def process_bdc_names(x):
        if "answer" in x and x["answer"]:
            # Use LLM to rewrite the text
            x["answer"] = (prompt | llm | StrOutputParser()).invoke({"text": x["answer"]})
        return x
    
    return RunnableLambda(process_bdc_names)



def create_main_chain(retriever, llm, guardian_llm, emb, vectorstore: VectorStore = None, retriever_top_k=5, score_threshold=0.5, compressor=None, hybrid_retriever=False):
    config = RailsConfig.from_path("config")
    
    
    # region: init guardrails
    guardrails = RunnableRails(config, 
                               llm=guardian_llm,
                               verbose=True, 
                               output_key="answer")
    # endregion: init guardrails    
    
    
    
    # region: init retriever
    if vectorstore is not None:
        print("using RetrieverWithScore")
        # retriever = RetrieverWithScore(vectorstore, search_type="similarity_score_threshold", search_kwargs={'score_threshold': score_threshold,'k':retriever_top_k})
        retriever = VectorStoreRetrieverWithScore(vectorstore, search_kwargs={'k':retriever_top_k})
    

    
    if hybrid_retriever:
        print("using hybrid retriever")
        emb_retriever_top_k = retriever_top_k//2
        
        documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(vectorstore.get()["documents"], vectorstore.get()["metadatas"])]
        
        # TODO: add similarity score to metadata
        bm25_retriever = BM25RetrieverWithScore.from_documents(documents = documents, 
                                                      k=retriever_top_k-emb_retriever_top_k, 
                                                      preprocess_func=word_tokenize, emb=emb)
        
        
        retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    
    if compressor is not None:
        print("using compressor (reranker)")
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
    # endregion: init retriever
    
    # region: create chains
    rag_chain = create_qa_rag_chain(retriever, llm)
    
    predefined_responses = load_yaml('./data/predefined_responses.yaml')

    predefined_response_chain = create_predefined_response_chain(predefined_responses, llm)
    
    
    
    response_branch = RunnableBranch(
        (lambda x: x["flag"] in ['d', 'a'], 
         lambda x: {**x, **rag_chain.invoke(x)}),
        (lambda x: x["flag"] == 'r', 
         lambda x: {**x, "answer": x["response"]}),  
         lambda x: {**x} # default case
    )
    


    
    main_chain = (predefined_response_chain 
                  | response_branch 
                  | create_bdc_response_regex_chain()
                #   | create_bdc_response_llm_chain(llm)
                  | RunnableLambda(merge_responses)
                  | RunnableLambda(fix_data_type))
    
    # main_chain.get_graph().print_ascii()
    
    return guardrails | main_chain
    










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
        

def create_time_filter(search_query: date_filter_params = None):
    
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














