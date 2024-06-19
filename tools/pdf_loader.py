import pdfplumber
from langchain_community.document_loaders import UnstructuredFileLoader
from typing import List
import os
from tools.text_spliter import ChineseTextSplitter

from langchain_community.document_loaders import UnstructuredPDFLoader

class UnstructuredPDFLoader(UnstructuredFileLoader):
  """Loader that uses unstructured to load image files, such as PNGs and JPGs."""

  def _get_elements(self) -> List:
    def pdf_ocr_txt(filepath, dir_path="tmp_files"):
      temp_txt_folder_path = os.path.join(os.path.dirname(filepath), dir_path)
      if not os.path.exists(temp_txt_folder_path):
        os.makedirs(temp_txt_folder_path)
      temp_txt_file_path = os.path.join(temp_txt_folder_path, 'temp.txt')
      with open(temp_txt_file_path, 'w', encoding='utf-8') as f:
        text = ""
        with pdfplumber.open(filepath) as pdf_reader:
          for page in pdf_reader.pages:
            text += page.extract_text(x_tolerance=1)
            f.write(page.extract_text())
      return temp_txt_file_path

    temp_txt_file_path = pdf_ocr_txt(self.file_path)
    from unstructured.partition.text import partition_text
    return partition_text(filename=temp_txt_file_path, **self.unstructured_kwargs)


if __name__ == "__main__":
    filepath = "/root/Code/ProjectRAG/content/samples/focal_loss.pdf"
    # loader = UnstructuredPDFLoader(filepath)
    loader = UnstructuredFileLoader(filepath, mode="elements")

    text_spliter = ChineseTextSplitter(pdf=True, sentence_size=100)
    docs = loader.load()
    for doc in docs:
      print(doc)
      print('\n')
      print(text_spliter.split_text1(doc.page_content))