# GLM4 RAG Agent

## 介绍
本项目基于vllm与langchain，使用GLM4实现了RAG与Agent工具调用，目前支持的工具有duckduckgo网络搜索。在对话时会优先从本地知识库中搜索相关信息，随后模型自动判断是否需要进一步调用工具来进行回答。

项目基于Fastapi实现openai类的LLM访问接口，基于langchain和faiss实现本地数据的解析与存储，参考glm4官方代码实现了自定义工具的解析与调用，可视化界面使用gradio实现。

## 快速上手

### 1. 环境配置

- 安装vllm（基于cuda12安装）
```angular2html
pip install vllm
```
基于cuda11.8安装vllm：
```angular2html
export VLLM_VERSION=0.4.3
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```
- 安装剩余项目依赖：

```angular2html
pip install -r requirements.txt
```

### 2. 模型下载

本项目使用[GLM4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)作为本地LLM模型，使用[bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)作为本地知识库的Embedding模型，模型可以从HuggingFace下载。

```angular2html
$ git lfs install
$ git clone https://huggingface.co/THUDM/glm-4-9b-chat
$ git clone https://huggingface.co/BAAI/bge-base-zh-v1.5
```

### 3. 确认模型参数配置
模型路径、知识库文档路径、本地知识库存储路径、api服务的ip和端口的配置可以在config文件夹中的model_config.py进行修改。

LLM模型相关的temperature、top_p参数以及知识库搜索相关参数可以在config文件夹中的base_config.py进行修改。

### 4. 启动LLM服务与界面服务
首先需要启动基于openai规则的LM服务：
```angular2html
$ python llm_api_server.py
```

随后启动对话界面服务：
```angular2html
$ python web_client_server.py
```

如果你不想使用可视化界面，项目还提供了命令行对话模式方便进行调试：
```angular2html
$ python agent_qa_demo.py
```