"""
Document Parser

This module provides the Docling/Marker PDF-to-Markdown parsing logic.
Per error_handling policies, parsing failures should be marked explicitly.
"""


class PDFParserTool:
    """
    Tool for converting PDF documents into structured text or markdown using Docling.
    """

    def parse_pdf(self, file_path: str) -> str:
        """
        Parses a PDF file from the local file system into markdown text using Docling.

        Args:
            file_path (str): The absolute path to the downloaded PDF file.

        Returns:
            str: The parsed markdown content.

        Raises:
            Exception: If the parsing process fails (fails loudly per .agrules).
        """
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            # Per .agrules: "PDF Parsing (Docling): If parsing fails, the paper must be marked 'FAILED' in State, not caught and skipped."
            # We raise so Analyst can handle and mark state as failed.
            raise Exception(f"Failed to parse PDF with Docling: {e}")
