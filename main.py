import io
import json
import os
import time

import dotenv
import gradio as gr
import requests
from PIL import Image
from googletrans import Translator

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


def translate(text):
    translator = Translator(timeout=10)
    return translator.translate(text, dest='en').text


def playground_v2(text):
    headers = {
        "Content-Type": "application/json",
        'Authorization': 'Token ' + os.getenv('REPLICATE_API_KEY'),
    }
    data = {
        "version": "42fe626e41cc811eaf02c94b892774839268ce1994ea778eba97103fe1ef51b8",
        "input": {
            "width": 1024,
            "height": 1024,
            "prompt": text,
            "scheduler": "K_EULER_ANCESTRAL",
            "guidance_scale": 3,
            "apply_watermark": False,
            "negative_prompt": "",
            "num_inference_steps": 50
        }
    }
    response = requests.post('https://api.replicate.com/v1/predictions', data=json.dumps(data), headers=headers)
    res = response.json()

    image_url = ''
    while True:
        time.sleep(0.5)
        response = requests.get(res['urls']['get'], headers=headers)
        res = response.json()
        if res['status'] == 'succeeded':
            image_url = res['output'][0]
            break
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    return img


def dosomething(text):
    return playground_v2(translate(text))


# 这里是主程序的代码
if __name__ == "__main__":
    # 加载配置
    dotenv.load_dotenv()
    ii = gr.Textbox(label='图片描述')
    oo = gr.Image(label='生成的图片')

    demo = gr.Interface(dosomething, inputs=ii, outputs=oo, title='基于playground-v2模型的图片生成',
                        allow_flagging='never')

    demo.launch(server_name='0.0.0.0', server_port=7862, share=True)
