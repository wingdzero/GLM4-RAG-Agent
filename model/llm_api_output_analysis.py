import json

from openai import OpenAI
from tools.function_tools.net_search import get_info_from_network
from tools.function_tools.calculater import calculate
from config.model_config import LOCAL_HOST, LOCAL_PORT
from config.base_config import TEMPERATURE, TOP_P

base_url = 'http://' + LOCAL_HOST + ':' + LOCAL_PORT + '/v1/'
client = OpenAI(api_key="EMPTY", base_url=base_url)

tools = [
        {
            "type": "function",
            "function": {
                "name": "get_info_from_network",
                "description": "在你无法回答问题时，从互联网搜索你不知道的事件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "你要查询的问题"
                        },
                    },
                    "required": ["question"],
                },
            }
        },
    {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "帮助你进行数学计算。计算按照先后顺序进行顺序计算，前面的计算用括号括起来。例子如下: a乘b表示为a*b，a的b次方表示为a**b，a除以b取余数表示为a%b，a的b次方加c除以d的余数表示为(a**b+c)%d，a的绝对值表示为abs(a)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equation": {
                            "type": "string",
                            "description": "将数学计算转换为描述里的指定计算符号"
                        },
                    },
                    "required": ["equation"],
                },
            }
        },
    ]

tool_names = ['get_info_from_network', 'calculate']


def function_chat(messages, use_stream=False):
    # 获得模型api的原始输出
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        tools=tools,
        stream=use_stream,
        max_tokens=256,
        temperature=TEMPERATURE,
        presence_penalty=1.2,
        top_p=TOP_P,
        tool_choice="auto"
    )
    if response:
        # 使用工具
        if response.choices[0].finish_reason == 'tool_calls':
            if use_stream:
                for chunk in response:
                    print(chunk)
            else:
                function_call = response.choices[0].message.tool_calls[0].function
                function_augments = json.loads(function_call.arguments)
                function_name = function_call.name
                # 解析网络搜索
                if function_name == 'get_info_from_network':
                    question = function_augments['question']
                    tool_result = get_info_from_network(question)
                    return {tool_result}
                elif function_name == 'calculate':
                    equation = function_augments['equation']
                    tool_result = calculate(equation)
                    print("GLM4: " + tool_result)
                    return tool_result
        # 结束自循环，输出回答
        elif response.choices[0].finish_reason == 'stop':
            if use_stream:
                for chunk in response:
                    print(chunk)
            else:
                output = response.choices[0].message.content
                print("GLM4: " + output)
                return output
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":

    messages = [
        {
            "role": "system",
            "content": "你将尽可能准确而专业地回答问题，不要回答多余的事情。",
        },
    ]

    while True:
        query = input("Input your question 请输入问题：").replace("\"", "")
        # local_knowledge = search_knowledgebase(query)
        # input = local_knowledge + query
        messages.append({"role": "user", "content": query})
        response = function_chat(messages)
        print(response)
        messages.append({"role": "assistant", "content": response})
        if len(messages) == 12:
            messages.pop(1)
            messages.pop(1)
