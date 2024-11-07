
from langchain_community.document_loaders import GitbookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from fastapi import FastAPI
from langserve import add_routes

from langchain.chains import RetrievalQA
from langchain.chains import create_history_aware_retriever
from langchain import hub





import streamlit as st
from langchain.chains import ConversationalRetrievalChain


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage

from langchain.callbacks.base import BaseCallbackHandler



from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

import json


# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing import Optional

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
# from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_community.query_constructors.chroma import ChromaTranslator


from babel.dates import format_date, format_datetime, format_time
from datetime import datetime, timedelta



from langchain.globals import set_debug
set_debug(False)



import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

from langchain_community.embeddings import OllamaEmbeddings



@st.cache_resource
def init_vars(retriever_top_k = 5):
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
        emb = OllamaEmbeddings(base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, temperature=0)
        print("model: ", EMBEDDING_MODEL, "base_url: ", EMBEDDING_URL)
    else:
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        print("model: text-embedding-3-small")

    

    vectorstore = Chroma(persist_directory=DB_PATH,embedding_function=emb)

    default_retriever = vectorstore.as_retriever(search_kwargs = {"k": retriever_top_k})

    return llm, emb, vectorstore, default_retriever, retriever_top_k



llm, emb, vectorstore, default_retriever, retriever_top_k = init_vars(retriever_top_k=5)




# print("test retriever: ", default_retriever.get_relevant_documents("RENCI"))



hasDate_prompt = ChatPromptTemplate.from_template(
"""
Find out if the user query has constraint on date or a range of time. Be careful, having date or time information or asking about time does not necessarily mean there is constraint on date. 

user query: {input}

Here are some examples: 

example_user: what happened yesterday
example_assistant: {{"hasDate":True}}

example_user: what happened in 2020 in Durham
example_assistant: {{"hasDate":True}}

example_user:when did 9/11 happen
example_assistant: {{"hasDate":False}}

example_user:american civil war started in 1861, tell me some fun facts about the civil war
example_assistant: {{"hasDate":False}}
"""
)


class QueryHasDate(BaseModel):
    hasDate: bool = Field(..., description="If the user query has time constraint.")

# hasDate_chain = hasDate_prompt | ChatOpenAI(temperature=0, model="gpt-4").with_structured_output(QueryHasDate)
hasDate_chain = hasDate_prompt | llm.with_structured_output(QueryHasDate)







class QueryDate(BaseModel):
    day: int = Field(..., description="The day of the contraint.",
                     ge=0, le=31)
    month: int = Field(..., description="The month of the contraint.",
                       ge=0, le=12)
    year: int = Field(..., description="The year of the contraint.")
    
class QueryDateRange(BaseModel):
    start: QueryDate = Field(..., description="The start date of the contraint.")
    end: QueryDate = Field(..., description="The end date of the contraint.")
    
    
daterange_prompt = ChatPromptTemplate.from_template(
f"""
Today is {format_date(datetime.now(), locale='en')}

Extract the date range ONLY for the constraint of the user query, return the start date and end date in the format of "day", "month", "year". Infer the missing information base on today's date. If the constraint has no exact date, set it to the first day of that period for the start date and the last day of that period for the end date. Set other missing information can't be inferred to 0. 

Having date information does not necessarily mean there is constraint on date. If there is no constraint on the day/month/year, return 0 for all fields.

Be careful with ambiguous date information, such as "last week", "yesterday", "a few days ago", etc.

Explain your reasoning first before giving the answer.

user query: {{input}}


Now, let's think step by step:
""")

# daterange_chain = daterange_prompt | ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(QueryDateRange)
daterange_chain = daterange_prompt | llm.with_structured_output(QueryDateRange)







class Search(BaseModel):
    start_date: Optional[int]
    end_date: Optional[int]

def construct_comparisons(query: Search):
    comparisons = []
    if query.start_date is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.GTE,
                attribute="timestamp",
                value=query.start_date,
            )
        )
    if query.end_date is not None:
        comparisons.append(
            Comparison(
                comparator=Comparator.LTE,
                attribute="timestamp",
                value=query.end_date,
            )
        )
    return comparisons



def date_filtered_retriever_constructor(metadata: QueryDateRange, vectorstore, top_k: int = 5, query: str = None):
    search_query = Search(start_date=datetime(metadata.start.year, metadata.start.month, metadata.start.day).timestamp(),
                          end_date=datetime(metadata.end.year, metadata.end.month, metadata.end.day, 23, 59, 59).timestamp())
    
    comparisons = construct_comparisons(search_query)
    print(comparisons)
    _filter = Operation(operator=Operator.AND, arguments=comparisons)
    rag_filter = ChromaTranslator().visit_operation(_filter)
    
    retriever = vectorstore.as_retriever(search_kwargs = {'filter': rag_filter, "k": top_k})

    
    
    # print(retriever.invoke(query))
    
    return retriever


def rag_chain_constructor(retriever):
    

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



    qa_system_prompt = """You are an assistant, called "RENCI Assistant", for question-answering tasks related RENCI. \
Use the following pieces of retrieved context to answer the question. \
If you can't get an answer base on the context, just say that you don't know. \
Use 1-3 paragraphs and keep the answer concise.\
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





default_rag_chain = rag_chain_constructor(default_retriever)










def parse_message(message) -> str:
    if answer:
        output = answer
    else:
        output = message["answer"]

    docs = message["context"]
    sources = []
    titles = []
    for doc in docs:
        source = doc.metadata["source"]
        if not source in sources:
            sources.append(source)
            if "title" in doc.metadata:
                titles.append(doc.metadata["title"])
            else: # if title not in metadata
                titles.append(f"{doc.metadata['doc_type']}: {doc.metadata['name']}")

    if len(sources) == 1:
        output += "\n\n#### Source:\n"
    elif len(sources) > 1:
        output += "\n\n#### Sources:\n"

    for i, source in enumerate(sources):
        output += f"{i + 1}. [{titles[i]}]({source})\n"

    return output




def parse_text(answer, context) -> str:

    output = answer
    docs = context
    
    sources = []
    titles = []
    ids = []
    for doc in docs:
        source = doc.metadata["source"]
        
        if not source in sources:
            sources.append(source)
            if "title" in doc.metadata and doc.metadata['doc_type'] != "person":
                titles.append(f"{doc.metadata['doc_type']}: {doc.metadata['title']}")
            else: # if title not in metadata
                titles.append(f"{doc.metadata['doc_type']}: {doc.metadata['name']}")

    if len(sources) == 1:
        output += "\n\n#### Source:\n"
    elif len(sources) > 1:
        output += "\n\n#### Sources:\n"

    for i, source in enumerate(sources):
        output += f"{i + 1}. [{titles[i]}]({source})\n"

    return output



# Set the title for the Streamlit app
st.title("RENCI GPT-3.5 Assistant")


def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)


# no streaming
def conversational_chat(query):
    # result = chain({"question": query, "chat_history": st.session_state['history']})
    
    if hasDate_chain.invoke({"input": query}).dict()['hasDate']:
        date_range = daterange_chain.invoke({"input": query})
        
        # if year is 0, set it to the current year
        if date_range.start.year == 0 or date_range.end.year == 0:
            date_range.start.year = datetime.now().year
            date_range.end.year = datetime.now().year
        
        # if month is 0, set it to all months
        if date_range.start.month == 0 or date_range.end.month == 0:
            date_range.start.month = 1
            date_range.end.month = 12
        
        # if day is 0, set it all days
        if date_range.start.day == 0:
            date_range.start.day = 1
            date_range.end.day = last_day_of_month(datetime(date_range.end.year, date_range.end.month, 1)).day
            

        
        print(date_range)
        filtered_retriever = date_filtered_retriever_constructor(date_range, vectorstore, top_k=retriever_top_k, query=query)
        filtered_rag_chain = rag_chain_constructor(filtered_retriever)
        
        result = filtered_rag_chain.invoke({"input": query, "chat_history": st.session_state['history']})
        
    else:
        print("No date constraint")
        result = default_rag_chain.invoke({"input": query, "chat_history": st.session_state['history']})
    
    
    st.session_state['history'].extend([HumanMessage(content=query), result["answer"]])
    return result



def select_chain(query):
    
    if hasDate_chain.invoke({"input": query}).dict()['hasDate']:
        date_range = daterange_chain.invoke({"input": query})
        
        # if year is 0, set it to the current year
        if date_range.start.year == 0 or date_range.end.year == 0:
            date_range.start.year = datetime.now().year
            date_range.end.year = datetime.now().year
        
        # if month is 0, set it to all months
        if date_range.start.month == 0 or date_range.end.month == 0:
            date_range.start.month = 1
            date_range.end.month = 12
        
        # if day is 0, set it all days
        if date_range.start.day == 0:
            date_range.start.day = 1
            date_range.end.day = last_day_of_month(datetime(date_range.end.year, date_range.end.month, 1)).day
        
        print(date_range)
        filtered_retriever = date_filtered_retriever_constructor(date_range, vectorstore, top_k=retriever_top_k, query=query)
        filtered_rag_chain = rag_chain_constructor(filtered_retriever)
        
        return filtered_rag_chain
    else:
        print("No date constraint")
        return default_rag_chain
    
    




# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'displayed_history' not in st.session_state:
    st.session_state['displayed_history'] = []



# Create containers for chat history and user input
response_container = st.container()
container = st.container()





if prompt := st.chat_input("Ask a question"):
    display_text = ""
    context = None
    # st.session_state.messages.append({"role": "user", "content": prompt})

    
    for i in range(len(st.session_state['displayed_history'])):
        role, content = st.session_state['displayed_history'][i]
        st.chat_message(role).write(content)
    
    
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.empty()

        
        
        container.markdown("Retrieving information...")
        
        current_chain = select_chain(prompt)        
        
        
        container.markdown("Thinking...")
        
        
        stream = current_chain.stream({"input": prompt, "chat_history": st.session_state['history']})

        for chunk in stream:
            if 'context' in chunk:
                context = chunk['context']
            
            if 'answer' in chunk:
                display_text += chunk['answer']
            container.markdown(display_text)
        
        
        # res = current_chain.invoke({"input": prompt, "chat_history": st.session_state['history']})
        # context = res["context"]
        # answer = res["answer"]
        # display_text += answer
        
        
        
        # display_text = parse_message(stream)
        answer = display_text
        display_text = parse_text(answer, context)
        container.markdown(display_text)


        
        # result = conversational_chat(prompt)
        # answer = result["answer"]
        # display_text = parse_text(answer, result["context"])
        
        # container.markdown(display_text)
        

        
    
    st.session_state['history'].extend([HumanMessage(content=prompt), answer])
    
    

    # st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state['displayed_history'].append(('user', prompt))
    st.session_state['displayed_history'].append(('assistant', display_text))


