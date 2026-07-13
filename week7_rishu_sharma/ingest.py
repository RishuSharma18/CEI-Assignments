import os


from utils.pdf_loader import load_pdf

from utils.text_splitter import split_documents

from utils.embeddings import get_embedding_model


from langchain_community.vectorstores import FAISS


from config import (
    DATA_PATH,
    VECTOR_DB_PATH
)



def create_vector_database():


    pdf_file = None


    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            pdf_file = os.path.join(
                DATA_PATH,
                file
            )

            break



    if pdf_file is None:

        raise Exception(
            "No PDF found inside data folder"
        )



    print("Loading PDF...")


    documents = load_pdf(
        pdf_file
    )



    print("Splitting text...")


    chunks = split_documents(
        documents
    )



    print(
        f"Created {len(chunks)} chunks"
    )



    print(
        "Creating embeddings..."
    )


    embeddings = get_embedding_model()



    print(
        "Saving vector database..."
    )



    db = FAISS.from_documents(

        chunks,

        embeddings

    )



    db.save_local(
        VECTOR_DB_PATH
    )


    print(
        "Vector database created successfully"
    )



if __name__=="__main__":

    create_vector_database()