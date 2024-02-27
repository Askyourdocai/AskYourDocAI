"""This script is used to retrieve relevant documents from a given pdf and query using chromadb."""

import os
import re
import time
from typing import Any
from typing import List
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from data_extractor import DataExtractor
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from qdrant_client import QdrantClient

load_dotenv()


class RetrieveRelevantDocsBase:
    """
    A base class for retrieving relevant documents from a given text.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        # Initialize the DataExtractor object
        self.pdf_extractor = DataExtractor(pdf_path)

    def get_preprocessed_text(self):
        """Returns the preprocessed text from the PDF file."""

        return self.pdf_extractor.get_preprocessed_text()

    def get_relevant_docs(self, query: str, n_results: int = 2) -> List[Any]:
        """
        Retrieves a list of relevant documents based on the query.

        Args:
            query (str): The query text.
            n_results (int): The number of results to return.

        Returns:
            List[Any]: A list of relevant documents.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def chunk_txtbytok_re(self, text: str, tokens_per_chunk: int = 1000):
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

    def chunk_txtbytok_txtsplitters(
        self, text: str, num_pages: int = 5, tokens_per_chunk: int = 1000
    ):
        """
        Splits the text into chunks based on the number of tokens using the txtsplitters.

        Args:
            text (str): The text to be split.
            tokens_per_chunk (int): The number of tokens per chunk.

        Returns:
            list: A list of chunks.
        """
        # Set chunk size based on number of pages
        if num_pages <= 3:
            chunk_size = 256
        elif 3 < num_pages <= 10:
            chunk_size = 500
        else:  # num_pages > 10
            chunk_size = 1000
        # Initialize the text splitters
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "], chunk_size=chunk_size, chunk_overlap=0
        )
        # Initialize the SentenceTransformersTokenTextSplitter
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=tokens_per_chunk
        )

        # Split the text into chunks of text
        character_split_texts = character_splitter.split_text(text)
        token_split_texts = []
        for text_chunk in character_split_texts:
            token_split_texts += token_splitter.split_text(text_chunk)
        return token_split_texts


class RetrieveRelevantDocsChromaDB(RetrieveRelevantDocsBase):
    """
    A class for retrieving relevant documents from a given text using chromadb.

    Attributes:

    Args:
        pdf_path (str): The path to the PDF file.

    Examples:
        >>> pdf_path = "path/to/pdf/file"
        >>> rrds = RetrieveRelevantDocsChromaDB(pdf_path)
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
        super().__init__(pdf_path)
        # Initialize the ChromaDB client
        self.chroma_client = chromadb.Client()
        self.chroma_collection: Optional[Any] = None
        # Initialize the SentenceTransformer embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction()

    def chunk_maker(self, text: str):
        """Returns a list of chunks of text.

        Args:
            text (str): The input text.

        Returns:
            list: A list of chunks of text.
        """
        # Store the number of pages when the DataExtractor is initialized
        num_pages = self.pdf_extractor.get_number_of_pages()
        # Split the text into chunks of text
        return super().chunk_txtbytok_txtsplitters(
            text, num_pages=num_pages, tokens_per_chunk=350
        )

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
        documents = self.chunk_maker("".join(super().get_preprocessed_text()))
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


class RetrieveRelevantDocsPinecone(RetrieveRelevantDocsBase):
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
        super().__init__(pdf_path)
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
        while not self.pc.describe_index(index_name).status["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(2)
        print("Index created successfully!")

    # Use the _chunk_text_by_tokens method from the base class for text chunking
    def chunk_maker(self, text: str):
        """
        Chunks the text into smaller pieces.

        Args:
            text (str): The input text.

        Returns:
            list: A list of chunks of text.
        """
        # Store the number of pages when the DataExtractor is initialized
        # num_pages = self.pdf_extractor.get_number_of_pages()
        # Split the text into chunks of text
        return super().chunk_txtbytok_re(text, tokens_per_chunk=500)

    def _preprocess_and_chunk_text(self):
        """
        Preprocesses the text and chunks it into smaller pieces.

        Returns:
            list: A list of chunks of text.
        """
        text_data = [super().get_preprocessed_text()]
        chunked_text_data = [
            chunk for text_block in text_data for chunk in self.chunk_maker(text_block)
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
            print("Index exists. making fassst...")
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


# pylint: disable=E1101
# pylint: disable=E1102
class RetrieveRelevantDocsQdrant(RetrieveRelevantDocsBase):
    """
    A class for retrieving relevant documents from a given text using Qdrant.

    Attributes:

    Args:
        pdf_path (str): The path to the PDF file.

    Examples:
        >>> pdf_path = "path/to/pdf/file"
        >>> rrdq = RetrieveRelevantDocsQdrant(pdf_path)
        >>> query = "What is the telephone number?"
        >>> retrieved_documents = rrdq.get_relevant_docs(query, n_results=2)
        >>> print("Query------------------:", query)
        >>> for document in retrieved_documents:
        >>>     print("====================================\n", document)
    """

    def __init__(self, pdf_path: str):
        """
        Initializes the RetrieveRelevantDocsQdrant object.

        Args:
            pdf_path (str): The path to the PDF file.

        """
        super().__init__(pdf_path)
        self.pdf_path = pdf_path
        self.embeddings = HuggingFaceEmbeddings()
        self.collection_name = self._generate_collection_name()
        self.qdrant_url = "https://9e48d23e-54a5-493e-af18-ff7a60a537c4.us-east4-0.gcp.cloud.qdrant.io:6333"
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.qdrant_client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key
        )

    def _generate_collection_name(self):
        """
        Generates a collection name from the PDF file name.

        Returns:
            str: The collection name.
        """
        base_name = os.path.basename(self.pdf_path)
        valid_name = re.sub(r"[^a-zA-Z]", "", base_name.split(".")[0]).lower()
        collection_name = valid_name[:5]
        return collection_name

    def chunk_maker(self, text: str):
        """
        Chunks the text into smaller pieces.

        Args:
            text (str): The input text.

        Returns:
            list: A list of chunks of text.
        """
        # Store the number of pages when the DataExtractor is initialized
        # num_pages = self.pdf_extractor.get_number_of_pages()
        # Split the text into chunks of text
        return super().chunk_txtbytok_re(text, tokens_per_chunk=100)

    def _preprocess_and_chunk_text(self):
        """
        Preprocesses the text and chunks it into smaller pieces.

        Returns:
            list: A list of chunks of text.
        """
        text_data = [super().get_preprocessed_text()]
        chunked_text_data = [
            chunk for text_block in text_data for chunk in self.chunk_maker(text_block)
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

    def _check_collection_exists(self):
        """
        Checks if the collection exists.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        if self.collection_name in [
            collection.name
            for collection in self.qdrant_client.get_collections().collections
        ]:
            return True
        return False

    def _create_collection(self):
        """
        Creates a collection in Qdrant.

        Returns:
            Qdrant: The Qdrant object.
        """
        print("Checking if collection exists...", self._check_collection_exists())
        documents = self._load_string_documents()
        vector_store = Qdrant.from_documents(
            documents,
            self.embeddings,
            url=self.qdrant_url,
            prefer_grpc=True,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
        )
        return vector_store

    def get_vector_store(self):
        """
        Returns the Qdrant vector store.

        Returns:
            Qdrant: The Qdrant object.
        """
        if self._check_collection_exists():
            print("Collection exists. Making fassst...")
            return Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )

        return self._create_collection()

    def get_relevant_docs(self, query: str, n_results: int = 2):
        """
        Returns a list of relevant documents.

        Args:
            query (str): The query text.
            n_results (int): The number of results to return.

        Returns:
            list: A list of relevant documents.
        """
        print("U r usng Qdrant")
        vector_store = self.get_vector_store()
        results = vector_store.max_marginal_relevance_search(
            query, k=n_results, fetch_k=10
        )
        return [[doc.page_content for doc in results]]
