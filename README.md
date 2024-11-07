# BDC Chatbot (WIP)
Chatbot for Biodata Catalyst


## Install dependencies
```bash
pip install -r requirements.txt
```


## Local development
```bash
kubectl -n ner port-forward svc/vllm-llama-3-1-8b-instruct 8080:80
kubectl -n ner port-forward svc/ollama 11434:11434
```

> [!WARNING]
> To use ${\color{orange}\text{vLLM}}$ API for chat completion, remove `parallel_tool_calls=False` in `langchain_openai\chat_models\base.py`

> [!IMPORTANT]
> Chroma DB initialization might quit without error or warning, might be caused by compatibility issue with Windows. 