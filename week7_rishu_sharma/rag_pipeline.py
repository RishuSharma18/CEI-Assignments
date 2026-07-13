from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embedding_model

from transformers import pipeline

from config import (
    VECTOR_DB_PATH,
    LLM_MODEL
)


class RAG:

    def __init__(self):

        print("Loading embedding model...")
        self.embeddings = get_embedding_model()

        print("Loading FAISS database...")
        self.db = FAISS.load_local(
            VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("Loading Language Model...")

        self.generator = pipeline(
            task="text-generation",
            model=LLM_MODEL,
            max_new_tokens=200,
            do_sample=False,
            truncation=True
        )

        print("RAG System Ready!")

    def retrieve(self, question):

        docs = self.db.similarity_search(
            question,
            k=1
        )

        return docs

    def generate_answer(self, question):

        docs = self.retrieve(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an AI assistant.

Answer ONLY from the given context.

If the answer is not present in the context, reply:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        result = self.generator(prompt)

        answer = result[0]["generated_text"]

        return answer, docs