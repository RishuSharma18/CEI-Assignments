import streamlit as st


from rag_pipeline import RAG



st.set_page_config(

    page_title="Document QA RAG",

    page_icon="📚"

)



st.title(
    "📚 Document Question Answering System"
)


st.write(
"""
Ask questions from your private documents.
Powered by Retrieval Augmented Generation.
"""
)



@st.cache_resource

def load_model():

    return RAG()



rag = load_model()



question = st.text_input(

    "Ask your question"

)



if question:


    answer, sources = rag.generate_answer(

        question

    )



    st.subheader(
        "Answer"
    )


    st.write(
        answer
    )



    st.subheader(
        "Retrieved Context"
    )



    for i,doc in enumerate(sources):


        with st.expander(
            f"Chunk {i+1}"
        ):

            st.write(
                doc.page_content
            )