import torch


MAX_MODEL_LENGTH = 8192

TEMPERATURE = 0.1

TOP_P = 0.1

# 文本分句长度
SENTENCE_SIZE = 100

# 匹配后单段上下文长度
CHUNK_SIZE = 250

# 知识库置信度分数（分数越低置信度越高）
KNOWLEDGE_CONFIDENCE_THRESHOLD = 1

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 5

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT_TEMPLATE = """已知信息：{context} 根据上述已知信息回答问题。不允许在答案中添加编造成分，答案请使用中文。请回答以下问题: """
