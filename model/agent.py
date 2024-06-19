import os

from tools.init_knowledge_base import init_knowledge_vector_store, search_knowledgebase
from model.llm_api_output_analysis import function_chat

from config.model_config import DOC_PATH, VS_PATH

class Agent:
    def __init__(self):
        self.doc_path = DOC_PATH
        self.vs_path = VS_PATH
        self.messages = [
                {
                    "role": "system",
                    "content": "你将尽可能准确而专业地回答问题。不要回答多余的事情。你可以使用工具进行网络搜索来回答问题。",
                },
            ]

    def init_agent(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        init_knowledge_vector_store(self.doc_path, self.vs_path, )


    def call(self, query):
        if query == '重新开始' or query == '重新开始对话':
            self.messages = [
                {
                    "role": "system",
                    "content": "你将尽可能准确而专业地回答问题。不要回答多余的事情。你可以使用工具进行网络搜索来回答问题。",
                },
            ]

        local_knowledge = search_knowledgebase(query)
        model_input = local_knowledge + query
        self.messages.append({"role": "user", "content": model_input})
        response = function_chat(self.messages)
        if isinstance(response, set):
            self.messages.append(
                {"role": "tool", "content": list(response)[0] + ' 基于以上信息判断是否能回答问题，并直接回答: ' + query})
            count = 0
            while True:
                response = function_chat(self.messages)
                if isinstance(response, set):
                    count += 1
                    if count == 3:
                        self.messages.append(
                            {"role": "tool",
                             "content": '工具无法给你提供帮助，请不要调用工具，直接回答此问题: ' + query})
                        break
                    else:
                        self.messages.append(
                            {"role": "tool",
                             "content": list(response)[0] + ' 基于以上信息判断是否能回答问题，并直接回答: ' + query})
                        continue
                else:
                    break

        self.messages.append({"role": "assistant", "content": response})
        if len(self.messages) == 15:
            self.messages.pop(1)
            self.messages.pop(1)

        return response
