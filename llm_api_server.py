import time
from asyncio.log import logger
import re
import uvicorn
import gc
import json
import torch

from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, LogitsProcessor
from sse_starlette.sse import EventSourceResponse
from config.model_config import LLM_MODEL_PATH, LOCAL_HOST, LOCAL_PORT
from config.base_config import MAX_MODEL_LENGTH

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

MODEL_PATH = LLM_MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = ["glm-4"]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: FunctionCall
    type: Literal["function"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = "None"
    repetition_penalty: Optional[float] = 1.1


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(output: str, use_tool: bool = False) -> Union[str, dict]:
    lines = output.strip().split("\n")
    arguments_json = None
    special_tools = ["cogview", "simple_browser"]

    tool_call_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    if len(lines) >= 2 and tool_call_pattern.match(lines[0]):
        function_name = lines[0].strip()
        arguments = "\n".join(lines[1:]).strip()

        try:
            arguments_json = json.loads(arguments)
            is_tool_call = True
        except json.JSONDecodeError:
            is_tool_call = function_name in special_tools

        if is_tool_call and use_tool:
            content = {
                "name": function_name,
                "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments, ensure_ascii=False)
            }
            if function_name == "simple_browser":
                search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                match = search_pattern.match(arguments)
                if match:
                    content["arguments"] = json.dumps({
                        "query": match.group(1),
                        "recency_days": int(match.group(2))
                    }, ensure_ascii=False)
            elif function_name == "cogview":
                content["arguments"] = json.dumps({
                    "prompt": arguments
                }, ensure_ascii=False)

            return content
    return output.strip()





@torch.inference_mode()
async def generate_stream_glm4(params):
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))

    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "repetition_penalty": repetition_penalty,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop_token_ids": [151329, 151336, 151338],
        "ignore_eos": False,
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        output_len = len(output.outputs[0].token_ids)
        input_len = len(output.prompt_token_ids)
        ret = {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            },
            "finish_reason": output.outputs[0].finish_reason,
        }
        yield ret
    gc.collect()
    torch.cuda.empty_cache()


def process_messages(messages, tools=None, tool_choice="none"):
    _messages = messages
    processed_messages = []
    msg_has_sys = False

    def filter_tools(tool_choice, tools):
        function_name = tool_choice.get('function', {}).get('name', None)
        if not function_name:
            return []
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    if tool_choice != "none":
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        if tools:
            processed_messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
            msg_has_sys = True

    if isinstance(tool_choice, dict) and tools:
        processed_messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        tool_calls = getattr(m, 'tool_calls', None)

        if role == "function":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "tool":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content,
                    "function_call": True
                }
            )
        elif role == "assistant":
            if tool_calls:
                for tool_call in tool_calls:
                    processed_messages.append(
                        {
                            "role": "assistant",
                            "metadata": tool_call.function.name,
                            "content": tool_call.function.arguments
                        }
                    )
            else:
                for response in content.split("\n"):
                    if "\n" in response:
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        metadata, sub_content = "", response
                    processed_messages.append(
                        {
                            "role": role,
                            "metadata": metadata,
                            "content": sub_content.strip()
                        }
                    )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            processed_messages.append({"role": role, "content": content})

    if not tools or tool_choice == "none":
        for m in _messages:
            if m.role == 'system':
                processed_messages.insert(0, {"role": m.role, "content": m.content})
                break
    return processed_messages



@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="glm-4")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")


    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        predict_stream_generator = predict_stream(request.model, gen_params)
        output = await anext(predict_stream_generator)
        if output:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
        logger.debug(f"First result output：\n{output}")

        function_call = None
        if output and request.tools:
            try:
                function_call = process_response(output, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)
            generate = parse_output_text(request.model, output, function_call=function_call)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

    response = ""
    async for response in generate_stream_glm4(gen_params):
        pass

    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    usage = UsageInfo()

    function_call, finish_reason = None, "stop"
    tool_calls = None
    if request.tools:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")

    if isinstance(function_call, dict):
        finish_reason = "tool_calls"
        function_call_response = FunctionCallResponse(**function_call)
        function_call_instance = FunctionCall(
            name=function_call_response.name,
            arguments=function_call_response.arguments
        )
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{int(time.time() * 1000)}",
                function=function_call_instance,
                type="function")]

    message = ChatMessage(
        role="assistant",
        content=None if tool_calls else response["text"],
        function_call=None,
        tool_calls=tool_calls,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        id="",
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )


async def predict_stream(model_id, gen_params):
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    function_name = None
    async for new_response in generate_stream_glm4(gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode
        lines = output.strip().split("\n")

        if not is_function_call and len(lines) >= 2 and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', lines[0]):
            is_function_call = True
            function_name = lines[0].strip()

        if is_function_call:
            for char in delta_text:
                function_call = {"name": function_name, "arguments": char}
                message = DeltaMessage(
                    content=None,
                    role="assistant",
                    function_call=function_call
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=None
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield chunk.model_dump_json(exclude_unset=True)
        else:
            if len(output) > 7:
                finish_reason = new_response.get("finish_reason", None)
                if not has_send_first_chunk:
                    message = DeltaMessage(
                        content="",
                        role="assistant",
                        function_call=None,
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=message,
                        finish_reason=finish_reason
                    )
                    chunk = ChatCompletionResponse(
                        model=model_id,
                        id="",
                        choices=[choice_data],
                        created=int(time.time()),
                        object="chat.completion.chunk"
                    )
                    yield chunk.model_dump_json(exclude_unset=True)

                send_msg = delta_text if has_send_first_chunk else output
                has_send_first_chunk = True
                message = DeltaMessage(
                    content=send_msg,
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield chunk.model_dump_json(exclude_unset=True)

    if is_function_call:
        yield json.dumps({"text": output})
    else:
        yield '[DONE]'


async def parse_output_text(model_id: str, value: str, function_call: FunctionCallResponse = None):
    delta = DeltaMessage(role="assistant", content=value)
    if function_call is not None:
        delta.function_call = function_call

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=delta,
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        # 占用显存的比例，请根据你的显卡显存大小设置合适的值，例如，如果你的显卡有80G，您只想使用24G，请按照24/80=0.3设置
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=False,
        engine_use_ray=False,
        disable_log_requests=True,
        max_model_len=MAX_MODEL_LENGTH,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host=LOCAL_HOST, port=int(LOCAL_PORT), workers=1)
