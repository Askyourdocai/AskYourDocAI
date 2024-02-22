"""This script retrieves relevant documents from a PDF using Pinecone embeddings."""

import os
import re

from create_pinecone_index import PineconeIndex
from data_extractor import DataExtractor
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone


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

    def upsert_documents(self):
        """
        Upserts the documents into the Pinecone index.

        Returns:
            Pinecone: The Pinecone object.
        """
        if self.index_name is None:
            self.index_name = self._generate_index_name()

        pinecone_index = PineconeIndex(self.index_name)
        docsearch = None

        if pinecone_index.check_index_exists():
            docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)
        else:
            pinecone_index.create_index()
            documents = self._load_string_documents()
            docsearch = Pinecone.from_documents(
                documents, self.embeddings, index_name=self.index_name
            )

        return docsearch

    def get_relevant_docs(self, query: str):
        """
        Returns a list of relevant documents.

        Args:
            query (str): The query text.

        Returns:
            list: A list of relevant documents.
        """
        docsearch = self.upsert_documents()
        results = docsearch.similarity_search(query)
        return results[0].page_content
