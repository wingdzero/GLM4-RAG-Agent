import os.path
import sys

# LLM模型路径
LLM_MODEL_PATH = '/root/Code/LLM_Models/glm-4-9b-chat'

# 本地知识库文档来源路径
DOC_PATH = '/root/Code/Chinese-LangChain-master/docs'

# 知识库embedding模型路径
EMBEDDING_MODEL = '/root/Code/RAG/bge-base-zh-v1.5'

# 知识库特征存储路径
VS_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'vector_store_library')

# LLM api服务ip
LOCAL_HOST = '127.0.0.1'
# LLM api服务端口号
LOCAL_PORT = '8000'
