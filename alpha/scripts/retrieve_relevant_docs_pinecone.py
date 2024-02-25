"""This script retrieves relevant documents from a PDF using Pinecone embeddings."""

import os
import re
import time

from data_extractor import DataExtractor
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

load_dotenv()


class RetrieveRelevantDocsPinecone:
    """
    A class for retrieving relevant documents from a given text using Pinecone.

    Attributes:

    Args:
        pdf_path (str): The path to the PDF file.

    Examples:
        >>> pdf_path = "path/to/pdf/file"
        >>> rrdsp = RetrieveRelevantDocsPinecone(pdf_path)
        >>> query = "What is the telephone number?"
        >>> retrieved_documents = rrdsp.get_relevant_docs(query, n_results=2)
        >>> print("Query------------------:", query)
        >>> for document in retrieved_documents:
        >>>     print("====================================\n", document)

    """

    def __init__(self, pdf_path: str):
        """
        Initializes the RetrieveRelevantDocsPinecone object.

        Args:
            pdf_path (str): The path to the PDF file.

        """
        # Store the path to the PDF file
        self.pdf_path = pdf_path
        # Initialize the HuggingFaceEmbeddings object
        self.embeddings = HuggingFaceEmbeddings()
        # Store the index name
        self.index_name = None
        # Initialize the Pinecone object
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)

    def _generate_index_name(self):
        """
        Generates an index name from the PDF file name.

        Returns:
            str: The index name.
        """
        # Extract the base name without extension and use first 7 characters
        base_name = os.path.basename(self.pdf_path)
        valid_name = re.sub(r"[^a-zA-Z]", "", base_name.split(".")[0]).lower()
        index_name = valid_name[:7]
        return index_name

    def _check_index_exists(self):
        """
        Checks if the index exists.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.index_name in self.pc.list_indexes().names()

    def _create_index(self, index_name: str):
        """
        Creates a Pinecone index.

        Returns:
            Pinecone: The Pinecone object.
        """
        spec = ServerlessSpec(cloud="aws", region="us-west-2")
        self.pc.create_index(index_name, dimension=768, metric="cosine", spec=spec)
        print(self.pc.describe_index(index_name).status["ready"])
        while not self.pc.describe_index(index_name).status["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(2)
        print("Index created successfully!")

    def _chunk_text_by_tokens(self, text: str, tokens_per_chunk: int = 1000):
        """
        Splits the text into chunks based on the number of tokens.

        Args:
            text (str): The text to be split.
            tokens_per_chunk (int): The number of tokens per chunk.

        Returns:
            list: A list of chunks.

        """
        # Split the text into tokens. Assuming tokens are words/punctuation separated by spaces
        tokens = re.findall(
            r"\S+|\s", text
        )  # This regex will separate words and spaces as tokens
        chunks = []
        current_chunk = []

        for token in tokens:
            current_chunk.append(token)
            # Check if the current chunk plus the next token exceeds the limit
            if len(current_chunk) >= tokens_per_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []

        # Add the last chunk if it contains any tokens
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _preprocess_and_chunk_text(self):
        """
        Preprocesses the text and chunks it into smaller pieces.

        Returns:
            list: A list of chunks of text.
        """
        text_data = [DataExtractor(self.pdf_path).get_preprocessed_text()]
        chunked_text_data = [
            chunk
            for text_block in text_data
            for chunk in self._chunk_text_by_tokens(text_block)
        ]
        return chunked_text_data

    def _load_string_documents(self):
        """
        Loads the string documents into the Document object.

        Returns:
            list: A list of Document objects.
        """
        documents = []
        chunked_text_data = self._preprocess_and_chunk_text()
        for i, chunk in enumerate(chunked_text_data):
            document = Document(page_content=chunk, metadata={"chunk_index": i})
            documents.append(document)
        return documents

    def _wait_for_documents_to_index(self):
        # Implement a heuristic waiting mechanism
        print("Waiting for documents to be indexed...")
        time.sleep(10)

    def upsert_documents(self):
        """
        Upserts the documents into the Pinecone index.

        Returns:
            Pinecone: The Pinecone object.
        """
        if self.index_name is None:
            self.index_name = self._generate_index_name()

        print("Checking if index exists...", self._check_index_exists())

        if self._check_index_exists():
            docsearch = PineconeVectorStore.from_existing_index(
                self.index_name, self.embeddings
            )
            return docsearch
        self._create_index(self.index_name)
        documents = self._load_string_documents()
        docsearch = PineconeVectorStore.from_documents(
            documents, self.embeddings, index_name=self.index_name
        )
        self._wait_for_documents_to_index()
        return docsearch

    def get_relevant_docs(self, query: str, n_results: int = 2):
        """
        Returns a list of relevant documents.

        Args:
            query (str): The query text.

        Returns:
            list: A list of relevant documents.
        """
        docsearch = self.upsert_documents()
        results = docsearch.similarity_search(query, k=n_results)
        reldocs = []
        for doc in results:
            reldocs.append(doc.page_content)

        return [reldocs]
