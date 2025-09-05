import os
import json
import yaml
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Authenticate with HuggingFace (required for gated LLaMA models)
# login(token=os.getenv("HF_TOKEN"))  # Or replace with your token directly

# Step 1: Load metadata from JSON/YAML files
def load_metadata_files(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                if filename.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
                text = json.dumps(data, indent=2)  # Convert to string for consistency
                documents.append(text)
    return documents

# Step 2: Prepare documents for RAG (split into chunks if large)
def prepare_documents(raw_texts):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for text in raw_texts:
        split_docs = splitter.create_documents([text])
        docs.extend(split_docs)
    return docs

# Step 3: Build vector store for retrieval
def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Step 4: Set up LLaMA 3.2 as the LLM
def setup_llm():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"  # Automatically maps to GPU/CPU
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        device_map="auto",
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm

# Step 5: Set up RAG chain with custom prompt
def setup_rag_chain(vector_store, llm):
    prompt_template = """
    You are an expert on dataset and model metadata. Use the following context to answer the question accurately.
    If the information isn't in the context, say "I don't have that information."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 chunks
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Main function to run the agent
def main():
    metadata_dir = "./metadata"  # Change to your directory
    raw_texts = load_metadata_files(metadata_dir)
    if not raw_texts:
        print("No metadata files found.")
        return
    
    docs = prepare_documents(raw_texts)
    vector_store = build_vector_store(docs)
    llm = setup_llm()
    rag_chain = setup_rag_chain(vector_store, llm)
    
    print("LLaMA 3.2 Agent ready. Ask questions about the metadata (type 'exit' to quit).")
    while True:
        query = input("Question: ")
        if query.lower() == 'exit':
            break
        response = rag_chain.run(query)
        print("Answer:", response)

if __name__ == "__main__":
    main()