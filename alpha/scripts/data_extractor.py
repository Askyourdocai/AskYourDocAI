"""Extracts and preprocesses text from PDF files."""

import re
from typing import List

import PyPDF2


class DataExtractor:
    """
    A class for extracting and preprocessing text from PDF files.

    Attributes:
        pdf_path (str): The path to the PDF file.
        text_list (list): A list to store preprocessed text from each page.

    Args:
        pdf_path (str): The path to the PDF file.

    Examples:
        >>> pdf_path = "path/to/pdf/file"
        >>> extractor = DataExtractor(pdf_path)
        >>> print(extractor.get_preprocessed_text())

    """

    def __init__(self, pdf_path: str):
        """
        Initializes the DataExtractor object.

        Args:
            pdf_path (str): The path to the PDF file.

        """
        self.pdf_path = pdf_path
        self.text_list: List[str] = []
        self.extract_text()

    def preprocess_text(self, text: str):
        """
        Preprocesses the input text by removing non-alphanumeric characters and extra whitespaces.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.

        """
        # Add your custom preprocessing steps here
        text = re.sub(r"[^a-zA-Z0-9\s,.\+@]", "", text)
        text = " ".join(text.split())

        # Remove text that might be part of logos or icons (customize as needed)
        text = self.remove_logo_text(text)

        return text

    def remove_logo_text(self, text: str):
        """
        Removes text that might be part of logos or icons based on predefined keywords.

        Args:
            text (str): The input text.

        Returns:
            str: The text with logo-related content removed.

        """
        # Add your logic to identify and remove logo text
        logo_keywords = ["logo", "icon", "copyright", "trademark"]
        for keyword in logo_keywords:
            text = re.sub(rf"\b{keyword}\b", "", text, flags=re.IGNORECASE)

        return text

    def extract_text(self):
        """
        Extracts text from the PDF file, preprocesses it, and appends to the text_list.

        """
        with open(self.pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Iterate through all pages in the PDF using enumerate instead of range
            for page_num, _ in enumerate(pdf_reader.pages):
                # Get the text from the current page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Preprocess the text
                text = self.preprocess_text(text)

                # Append the preprocessed text to the list
                self.text_list.append(text)

    def get_number_of_pages(self):
        """
        Returns the number of pages in the PDF file.

        Returns:
            int: The number of pages in the PDF file.

        """
        return len(self.text_list)

    def get_preprocessed_text(self):
        """
        Returns the preprocessed text as a single string.

        Returns:
            str: The preprocessed text.

        """
        return "\n".join(self.text_list)
