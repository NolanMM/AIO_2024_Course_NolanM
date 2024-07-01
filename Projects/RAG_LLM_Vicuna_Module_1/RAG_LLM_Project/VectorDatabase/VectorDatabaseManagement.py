import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class PDFRetrievalSystem:
    """
    PDFRetrievalSystem Class

    The PDFRetrievalSystem class provides a streamlined and efficient method for loading, processing,
    embedding, and retrieving information from a collection of PDF documents. It utilizes advanced 
    natural language processing techniques and vector databases to ensure accurate and relevant 
    information retrieval.

    Attributes:
    -----------
    folder_path : str
        The path to the folder containing the PDF documents.
    all_documents : list
        A list to hold all document texts.
    embedding : HuggingFaceEmbeddings
        The HuggingFace embeddings model.
    vector_db : Chroma
        The vector database created from the documents.
    retriever : Retriever
        The retriever for querying the vector database.
    """

    def __init__(self, folder_path='./VectorDatabase/Datasets'):
        """
        Initializes the PDFRetrievalSystem with the specified folder path and sets up the language model pipeline.

        Parameters:
        -----------
        folder_path : str
            The path to the folder containing the PDF documents.
        """
        self.folder_path = folder_path
        self.all_documents = []
        self.embedding = HuggingFaceEmbeddings()
        self.vector_db = None
        self.retriever = None

    def load_documents(self):
        """
        Loads and processes all PDF documents in the specified folder.

        This method iterates through all files in the folder path, loads each PDF using PyPDFLoader,
        splits the text into chunks, and stores the chunks in the all_documents list.
        """
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(self.folder_path, filename))
                text = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100)
                docs = splitter.split_documents(text)
                self.all_documents.extend(docs)

    def create_vector_db(self):
        """
        Creates a vector database from the loaded documents using the HuggingFace embeddings.

        This method initializes the vector database (Chroma) and converts it into a retriever.
        """
        self.vector_db = Chroma.from_documents(
            documents=self.all_documents, embedding=self.embedding)
        self.retriever = self.vector_db.as_retriever()

    def retrieve(self, query):
        """
        Retrieves relevant information based on the provided query and uses the language model to generate an answer.

        Parameters:
        -----------
        query : str
            The query or question to be retrieved.

        Returns:
        --------
        result : str
            The relevant information retrieved from the vector database.
        answer : str
            The generated answer from the language model.

        Raises:
        -------
        ValueError:
            If the retriever is not initialized.
        """
        if self.retriever is None:
            raise ValueError(
                "Retriever is not initialized. Call create_vector_db() first.")

        # Retrieve relevant documents
        docs = self.retriever.invoke(query)
        return docs
