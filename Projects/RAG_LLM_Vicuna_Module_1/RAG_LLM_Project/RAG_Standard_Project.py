from LLM_Models.VicunaLLM import vicuna_llm_model_implement
from VectorDatabase.VectorDatabaseManagement import PDFRetrievalSystem


def retrieve_answer_vicuna_model(query, docs):
    vicuna_model = vicuna_llm_model_implement()
    model_pipeline = vicuna_model.model_pipeline
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)

    # Use the language model to generate an answer
    input_text = f"Context: {formatted_docs}\n\nQuestion: {query}\n\nAnswer:"
    generated_text = model_pipeline(input_text)
    answer = generated_text[0]['generated_text'].split('Answer:')[
        1].strip()

    return formatted_docs, answer


if __name__ == "__main__":
    query = "What is Scala?"
    pdf_system = PDFRetrievalSystem()
    pdf_system.load_documents()
    pdf_system.create_vector_db()
    docs_ = pdf_system.retrieve(query)
    context, answer = retrieve_answer_vicuna_model(query, docs_)
    # print(f"Context: {context}")
    print(f"Answer: {answer}")
