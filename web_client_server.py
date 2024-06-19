import gradio as gr

from model.agent import Agent

agent = Agent()
agent.init_agent()

def echo(message, history):
    response = agent.call(message)
    # print(history)
    return response

demo = gr.ChatInterface(fn=echo, examples=["你是谁", "你可以做什么"], title="GLM4 Agent")
demo.launch(share=True)