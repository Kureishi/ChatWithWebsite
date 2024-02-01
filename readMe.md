##### Purpose:
This application allows any website (with informational text) to be passed in and for queries to be ran against the information retrieved. A RAG pipeline is implemented since the LLM was most likely not trained on the data. This way it can still generate responses from data it hasn't seen by vectorizing them, then retrieving relevant pieces of data pertaining to the query.

##### Notes:
- LLM: gpt4all-falcon-newbpe-q4_0.gguf
- Embeddings Model: hkunlp/instructor-xl
- Primary Library: Langchain 0.1.0

##### Caution:
Since application uses a local LLM model (from GPT4All where n_ctx set), the number of tokens may exceed the allowed context window when querying and a LLaMa ERROR may occur. Note the embeddings model used is from HuggingFace and another model may yield better results.

Different results may be obtained if another embeddings model and/or LLM are used (ex: OpenAI).