"""This script is used to retrieve relevant documents from a given pdf and query."""

import os
import re
from typing import Any
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from data_extractor import DataExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter


class RetrieveRelevantDocs:
    """
    A class for retrieving relevant documents from a given text.

    Attributes:

    Args:
        pdf_path (str): The path to the PDF file.

    Examples:
        >>> pdf_path = "path/to/pdf/file"
        >>> rrds = RetrieveRelevantDocs(pdf_path)
        >>> query = "What is the telephone number?"
        >>> retrieved_documents = rrds.get_relevant_docs(query, n_results=2)
        >>> print("Query------------------:", query)
        >>> for document in retrieved_documents:
        >>>     print("====================================\n", document)

    """

    def __init__(self, pdf_path: str):
        """
        Initializes the RetrieveRelevantDocs object.

        Args:
            pdf_path (str): The path to the PDF file.

        """
        # Store the path to the PDF file
        self.pdf_path = pdf_path
        # Initialize the DataExtractor object
        self.pdf_extractor = DataExtractor(pdf_path)
        # Initialize the ChromaDB client
        self.chroma_client = chromadb.Client()
        self.chroma_collection: Optional[Any] = None
        # Initialize the SentenceTransformer embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        # Store the number of pages when the DataExtractor is initialized
        num_pages = self.pdf_extractor.get_number_of_pages()

        # Set chunk size based on number of pages
        if num_pages <= 3:
            chunk_size = 256
        elif 3 < num_pages <= 10:
            chunk_size = 500
        else:  # num_pages > 10
            chunk_size = 1000

        # Initialize the text splitters
        self.character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "], chunk_size=chunk_size, chunk_overlap=0
        )
        # Initialize the SentenceTransformersTokenTextSplitter
        self.token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=256
        )

    def get_preprocessed_text(self):
        """Returns the preprocessed text from the PDF file."""

        return self.pdf_extractor.get_preprocessed_text()

    def chunk_maker(self, text: str):
        """Returns a list of chunks of text.

        Args:
            text (str): The input text.

        Returns:
            list: A list of chunks of text.
        """
        # Split the text into chunks of text
        character_split_texts = self.character_splitter.split_text(text)
        token_split_texts = []
        for text_chunk in character_split_texts:
            token_split_texts += self.token_splitter.split_text(text_chunk)
        return token_split_texts

    def create_collection(self):
        """Creates a collection in ChromaDB."""
        base_name = os.path.basename(self.pdf_path)
        valid_name = re.sub(r"[^a-zA-Z]", "", base_name.split(".")[0])
        collection_name = valid_name[:63]
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            collection_name, embedding_function=self.embedding_function
        )

    def add_documents_to_collection(self):
        """Adds documents to the collection in ChromaDB."""
        documents = self.chunk_maker("".join(self.get_preprocessed_text()))
        ids = [str(i) for i in range(len(documents))]
        # Add documents in batches for efficiency
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.chroma_collection.add(
                ids=ids[i : i + batch_size], documents=documents[i : i + batch_size]
            )

    def query_collection(self, query: str, n_results: int):
        """Queries the collection in ChromaDB.

        Args:
            query (str): The query text.
            n_results (int): The number of results to return.

        Returns:
            list: A list of relevant documents.
        """
        if self.chroma_collection is None:
            self.create_collection()
        assert self.chroma_collection is not None

        return self.chroma_collection.query(query_texts=[query], n_results=n_results)[
            "documents"
        ]

    def get_relevant_docs(self, query: str, n_results: int = 2):
        """Returns a list of relevant documents.

        Args:
            query (str): The query text.
            n_results (int): The number of results to return.

        Returns:
            list: A list of relevant documents.
        """
        if self.chroma_collection is None:
            self.create_collection()
        self.add_documents_to_collection()
        return self.query_collection(query, n_results)
