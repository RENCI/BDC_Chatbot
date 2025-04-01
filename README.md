# BDC Chatbot (WIP)

Chatbot for Biodata Catalyst

## Get documentation
Clone the [BDC website repo](https://github.com/stagecc/interim-bdc-website/tree/main) to a directory with the same parent directory as this repo.

## Copy environment variables
Copy `.env_example` to `.env` and make any necessary changes

## Install dependencies

```bash
pip install -r requirements.txt
```

## Port forwarding for local development using vLLM and Ollama at RENCI

```bash
kubectl -n ner port-forward svc/vllm-llama-3-1-8b-instruct 8080:80  # vLLM port-forward

kubectl -n ner port-forward svc/ollama 11434:11434                  # Ollama port-forward
```
Edit `.env` to match

## Create RAG database

```bash
python -m utils.prepare_preproc_doc                 # preprocess BDC website repo
python -m utils.prepare_chromadb                    # create chroma db
```

## Run chatbot
### Standalone Streamlit App
```bash
streamlit run app.py                                # run streamlit app
```

### Streamlit Frontend and LangServe/FastAPI Backend
```bash
fastapi dev server.py                               # run langserve backend
streamlit run client.py                             # run streamlit frontend
```

### Example Return
```json
{
    "input": user query,
    "chat_history": chat history,
    "context": retrieved docs,
    "topic": query topic (for predefined response),
    "response": predefined response,
    "flag": predefined response flag,
    "answer": chatbot answer,
    "display_answer": chatbot answer with predefined response,
    "bdcbot_response": bdcbot answer (empty if not called),
    "dugbot_response": dugbot answer (empty if not called)
}
```

> [!WARNING]
>
> - If you see `ValueError: Received disallowed comparator nin` when running the chatbot app, add `Comparator.IN, Comparator.NIN` to `langchain_community\query_constructors\chroma.py` under `allowed_comparators`

> [!IMPORTANT]
>
> - To use ${\color{orange}\text{vLLM}}$ API for chat completion, remove `parallel_tool_calls=False` from `langchain_openai\chat_models\base.py`
> - Chroma DB initialization might quit without error or warning, might be caused by compatibility issue with Windows.
