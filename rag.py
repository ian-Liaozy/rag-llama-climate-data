import os
import time
import json
import numpy as np
from tqdm import tqdm
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ----- Load and split documents -----
def load_and_chunk_documents(data_dir):
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, filename))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)


# ----- Create FAISS vector index -----
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    return db


# ----- Load retriever -----
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings)


# ----- Load LLM (Replace with LLaMA-3B on HPC as needed) -----
def load_llm():
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)


# ----- Apply similarity-based reranking -----
def rerank_docs(query: str, docs: List[str], embedding_model):
    query_emb = embedding_model.embed_query(query)
    doc_scores = [(doc.page_content, np.dot(query_emb, embedding_model.embed_documents([doc.page_content])[0])) for doc in docs]
    return sorted(doc_scores, key=lambda x: x[1], reverse=True)


# ----- Build RAG QA pipeline -----
def build_rag_pipeline():
    retriever = load_vector_store().as_retriever(search_type="similarity", k=6)
    llm = load_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# ----- Evaluate system performance -----
def evaluate_rag_system(eval_path="eval/questions.jsonl", rerank=False):
    with open(eval_path) as f:
        examples = [json.loads(line) for line in f]

    rag = build_rag_pipeline()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    total_time, correct = 0, 0

    for ex in tqdm(examples, desc="Evaluating RAG"):
        query = ex["question"]
        expected = ex["answer"]

        start = time.time()
        if rerank:
            raw_docs = rag.retriever.get_relevant_documents(query)
            top_reranked = rerank_docs(query, raw_docs, embedding_model)
            context = " ".join([doc[0] for doc in top_reranked[:3]])
            prompt = f"Answer based on: {context}\n\nQuestion: {query}\nAnswer:"
            answer = rag.llm(prompt)
        else:
            answer = rag.run(query)
        elapsed = time.time() - start

        total_time += elapsed
        if expected.lower() in answer.lower():
            correct += 1

        print(f"\nQ: {query}\nExpected: {expected}\nPredicted: {answer}\n---")

    accuracy = correct / len(examples)
    avg_time = total_time / len(examples)

    print("\n\n===== Evaluation Summary =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Latency: {avg_time:.3f}s per request")


# ----- Optional: Mock fine-tuned model baseline -----
def evaluate_finetuned_baseline(eval_path="eval/questions.jsonl"):
    with open(eval_path) as f:
        examples = [json.loads(line) for line in f]

    # Simulate a fast response (mock baseline)
    total_time, correct = 0, 0
    for ex in tqdm(examples, desc="Evaluating Fine-Tuned Baseline"):
        query, expected = ex["question"], ex["answer"]
        start = time.time()
        # TODO: Replace with actual fine-tuned model call
        answer = "mock_answer"
        time.sleep(0.1)  # simulate latency
        elapsed = time.time() - start

        total_time += elapsed
        if expected.lower() in answer.lower():
            correct += 1

    print("\n\n===== Fine-Tuned Model Baseline =====")
    print(f"Accuracy: {correct / len(examples) * 100:.2f}%")
    print(f"Average Latency: {total_time / len(examples):.3f}s per request")


# ----- CLI Entry -----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interactive", "rag_eval", "finetune_eval"], default="interactive")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("faiss_index"):
        chunks = load_and_chunk_documents("data/")
        build_vector_store(chunks)

    if args.mode == "interactive":
        rag = build_rag_pipeline()
        while True:
            query = input("Your Question (or 'exit'): ")
            if query.lower() == "exit":
                break
            print("Answer:", rag.run(query))
    elif args.mode == "rag_eval":
        evaluate_rag_system(rerank=args.rerank)
    elif args.mode == "finetune_eval":
        evaluate_finetuned_baseline()
