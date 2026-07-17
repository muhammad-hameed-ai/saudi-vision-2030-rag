import os
import json
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="saudi_vision_2030",
        url="http://localhost:6333",
        vector_name="dense",
    )


def retrieve_context(store, query, k=3):
    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = (
        "You are an expert analyst on Saudi Vision 2030 policy documents.\n"
        "Answer the question using ONLY the context provided below.\n"
        "If the answer is not in the context, say: "
        "I cannot find this information in the provided documents.\n\n"
        "CONTEXT:\n" + context_text
        + "\n\nQUESTION:\n" + question
        + "\n\nANSWER:"
    )
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'num_ctx': 2048, 'num_predict': 256}
    )
    return response['message']['content']


def score_faithfulness(answer, contexts):
    context_text = "\n\n".join(contexts)
    prompt = (
        "Given the following context and answer, rate how faithful the answer "
        "is to the context on a scale of 0.0 to 1.0.\n"
        "1.0 means every claim in the answer is supported by the context.\n"
        "0.0 means the answer contains claims not found in the context.\n"
        "Reply with ONLY a number between 0.0 and 1.0, nothing else.\n\n"
        "CONTEXT:\n" + context_text
        + "\n\nANSWER:\n" + answer
        + "\n\nSCORE:"
    )
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'num_ctx': 2048, 'num_predict': 10}
    )
    try:
        return float(response['message']['content'].strip().split()[0])
    except Exception:
        return 0.5


def score_relevancy(question, answer):
    prompt = (
        "Given the following question and answer, rate how relevant the answer "
        "is to the question on a scale of 0.0 to 1.0.\n"
        "1.0 means the answer directly and completely addresses the question.\n"
        "0.0 means the answer is completely off-topic.\n"
        "Reply with ONLY a number between 0.0 and 1.0, nothing else.\n\n"
        "QUESTION:\n" + question
        + "\n\nANSWER:\n" + answer
        + "\n\nSCORE:"
    )
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'num_ctx': 1024, 'num_predict': 10}
    )
    try:
        return float(response['message']['content'].strip().split()[0])
    except Exception:
        return 0.5


def score_context_precision(question, contexts):
    scores = []
    for ctx in contexts:
        prompt = (
            "Given the following question and context chunk, rate how relevant "
            "this chunk is for answering the question on a scale of 0.0 to 1.0.\n"
            "Reply with ONLY a number between 0.0 and 1.0, nothing else.\n\n"
            "QUESTION:\n" + question
            + "\n\nCONTEXT CHUNK:\n" + ctx
            + "\n\nSCORE:"
        )
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'num_ctx': 1024, 'num_predict': 10}
        )
        try:
            scores.append(float(response['message']['content'].strip().split()[0]))
        except Exception:
            scores.append(0.5)
    return sum(scores) / len(scores) if scores else 0.0


eval_questions = [
    "What are the main economic goals of Saudi Vision 2030?",
    "What role does the private sector play in Vision 2030?",
    "What is the Public Investment Fund and what is its role?",
    "What are the Vision Realization Programs?",
    "How does Vision 2030 aim to develop the entertainment sector?",
]


def main():
    print("Loading vector store...")
    store = get_vector_store()
    print("Running evaluation on 5 questions...\n")

    all_faithfulness = []
    all_relevancy = []
    all_precision = []
    results_log = []

    for i, question in enumerate(eval_questions):
        print(f"Question {i+1}/5: {question[:60]}...")

        contexts = retrieve_context(store, question, k=3)
        answer = generate_answer(question, contexts)

        faith = score_faithfulness(answer, contexts)
        relev = score_relevancy(question, answer)
        prec = score_context_precision(question, contexts)

        all_faithfulness.append(faith)
        all_relevancy.append(relev)
        all_precision.append(prec)

        results_log.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": prec
        })

        print(f"  Faithfulness: {faith:.2f} | "
              f"Relevancy: {relev:.2f} | "
              f"Precision: {prec:.2f}")

    avg_faith = sum(all_faithfulness) / len(all_faithfulness)
    avg_relev = sum(all_relevancy) / len(all_relevancy)
    avg_prec = sum(all_precision) / len(all_precision)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Faithfulness      : {avg_faith:.4f}")
    print(f"Answer Relevancy  : {avg_relev:.4f}")
    print(f"Context Precision : {avg_prec:.4f}")
    print("=" * 60)

    os.makedirs("data/evaluation", exist_ok=True)
    final_scores = {
        "faithfulness": round(avg_faith, 4),
        "answer_relevancy": round(avg_relev, 4),
        "context_precision": round(avg_prec, 4),
        "per_question": results_log
    }

    with open("data/evaluation/evaluation_scores.json", "w") as f:
        json.dump(final_scores, f, indent=2)

    print("\nFull results saved to data/evaluation/evaluation_scores.json")


if __name__ == "__main__":
    main()