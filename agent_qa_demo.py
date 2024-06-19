import os
import sys

from tools.init_knowledge_base import search_knowledgebase, init_knowledge_vector_store
from model.llm_api_output_analysis import function_chat

def main():

    va_path = os.path.join(os.path.dirname(sys.argv[0]), 'vector_store_library')
    if not os.path.exists(va_path):
        os.makedirs(va_path)
    file_path = '/root/Code/Chinese-LangChain-master/docs'
    init_knowledge_vector_store(file_path, va_path, )

    messages = [
        {
            "role": "system",
            # "content": "你将尽可能准确而专业地回答问题。不要回答多余的事情。",
            "content": "你将尽可能准确而专业地回答问题。不要回答多余的事情。你可以使用工具进行网络搜索来回答问题。",  # 6月17日修改
        },
    ]

    while True:
        query = input("请输入: ").replace("\"", "")
        if query == '重新开始' or query == '重新开始对话':
            messages = [
                {
                    "role": "system",
                    "content": "你将尽可能准确而专业地回答问题。不要回答多余的事情。你可以使用工具进行网络搜索来回答问题。",
                },
            ]
        local_knowledge = search_knowledgebase(query)
        model_input = local_knowledge + query
        messages.append({"role": "user", "content": model_input})
        response = function_chat(messages)
        if isinstance(response, set):
            messages.append({"role": "tool", "content": list(response)[0] + ' 基于以上信息判断是否能回答问题，并直接回答: ' + query})
            count = 0
            while True:
                response = function_chat(messages)
                if isinstance(response, set):
                    count += 1
                    if count == 3:
                        messages.append(
                            {"role": "tool",
                             "content": '工具无法给你提供帮助，请不要调用工具，直接回答此问题: ' + query})
                        break
                    else:
                        messages.append(
                            {"role": "tool",
                             "content": list(response)[0] + ' 基于以上信息判断是否能回答问题，并直接回答: ' + query})
                        continue
                else:
                    break
        # print(response)
        messages.append({"role": "assistant", "content": response})
        if len(messages) == 15:
            messages.pop(1)
            messages.pop(1)

if __name__ == '__main__':
    main()