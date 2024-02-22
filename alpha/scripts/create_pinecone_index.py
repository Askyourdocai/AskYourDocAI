""" This script creates a Pinecone index and provides methods to interact with the index."""

import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec

load_dotenv()


class PineconeIndex:
    """
    A class for creating a Pinecone index and interacting with it.

    Attributes:

    Args:
        index_name (str): The name of the index.
    """

    def __init__(self, index_name: str):
        """
        Initializes the PineconeIndex object.

        Args:
            index_name (str): The name of the index.
        """
        self.index_name = index_name
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)

    def create_index(self):
        """
        Creates a Pinecone index.

        Returns:
            None
        """
        spec = ServerlessSpec(cloud="aws", region="us-west-2")
        self.pc.create_index(self.index_name, dimension=768, metric="cosine", spec=spec)
        while not self.pc.describe_index(self.index_name).status["ready"]:
            time.sleep(1)

    def check_index_exists(self):
        """
        Checks if the index exists.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.index_name in self.pc.list_indexes().names()

    def delete_index(self):
        """
        Deletes the index.

        Returns:
            None
        """
        self.pc.delete_index(self.index_name)
