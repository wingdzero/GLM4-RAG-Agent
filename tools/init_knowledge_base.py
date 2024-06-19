import os
import datetime

from logger import logger
from tqdm import tqdm
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader, UnstructuredWordDocumentLoader, \
    UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from config.model_config import VS_PATH, EMBEDDING_MODEL
from config.base_config import EMBEDDING_DEVICE, PROMPT_TEMPLATE, SENTENCE_SIZE
from tools.text_spliter import ChineseTextSplitter
from tools.pdf_loader import UnstructuredPDFLoader
from pypinyin import lazy_pinyin
from config.base_config import KNOWLEDGE_CONFIDENCE_THRESHOLD


def tree(directory):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    ret_list = []
    basename_list = []
    if isinstance(directory, str):
        if not os.path.exists(directory):
            print("路径不存在")
            return None, None
        for root, dirs, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                ret_list.append(full_path)
                basename_list.append(os.path.basename(full_path))
    return ret_list, basename_list


def load_file(filepath: str, sentence_size=100):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        docs = loader.load()
        # textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        # docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".doc") or filepath.lower().endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".ppt") or filepath.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    # elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
    #     loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
    #     textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
    #     docs = loader.load_and_split(text_splitter=textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    # write_check_file(filepath, docs)  # 记录成功保存了多少文件
    return docs


def init_knowledge_vector_store(filepath, vs_path):
    global vector_store
    loaded_files = []
    failed_files = []
    docs = []

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs={'device': EMBEDDING_DEVICE})

    if vs_path and os.path.exists(vs_path) and "index.faiss" in os.listdir(vs_path):
        logger.info("检测到已有数据库，正在加载")
        vector_store = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("已有数据库加载成功")

    if not os.path.exists(filepath):
        print("文件路径不存在")
        return None

    elif os.path.isdir(filepath):
        for fullfilepath, file in tqdm(zip(*tree(filepath)), desc="加载文件"):
            try:
                docs += load_file(fullfilepath, SENTENCE_SIZE)
                loaded_files.append(fullfilepath)
            except Exception as e:
                logger.error(e)
                failed_files.append(file)

        if len(failed_files) > 0:
            logger.info("以下文件未能成功加载：")
            for file in failed_files:
                logger.info(f"{file}\n")

    if len(docs) > 0:
        logger.info("文件加载完毕，正在生成向量库")
        if vs_path and os.path.exists(vs_path) and "index.faiss" in os.listdir(vs_path):
            logger.info("基于已有数据库导入文件")
            vector_store.add_documents(docs)
        else:
            if not vs_path:
                vs_path = os.path.join(VS_PATH,
                                       f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
                if not os.path.exists(vs_path):
                    os.makedirs(vs_path)
            logger.info("未检测到已有数据库，正在生成新的向量库")
            vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(vs_path)


def search_knowledgebase(query):
    related_docs_with_score = vector_store.similarity_search_with_score(query, k=1)

    context = "\n".join([doc[0].page_content for doc in related_docs_with_score])
    prompt = PROMPT_TEMPLATE.replace("{context}", context)
    if related_docs_with_score[0][1] <= KNOWLEDGE_CONFIDENCE_THRESHOLD:
        print(f"参考资料选自: {related_docs_with_score[0][0].metadata}, 置信度为: {related_docs_with_score[0][1]}")
        return prompt
    else:
        return ""
