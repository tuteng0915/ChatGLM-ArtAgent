import gradio as gr
from utils import *
import requests
import os
import io
import base64
from PIL import Image
import time
import random
import json
import shutil

os.makedirs('output', exist_ok=True)

# 定义新的函数
def call_sd_t2i(pos_prompt, neg_prompt, user_input=""):
    seed = random.randint(0, 1000000)  # 添加随机种子
    url = "http://127.0.0.1:6016"
    payload = {
        "enable_hr": True,
        "denoising_strength": 0.55,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "prompt": pos_prompt,
        "steps": sd_steps,
        "negative_prompt": neg_prompt,
        "cfg_scale": sd_cfg,
        "batch_size": 1,
        "n_iter": 1,
        "width": sd_width,
        "height": sd_height,
        "seed": seed
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    image_list = []
    os.makedirs('output', exist_ok=True)
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        image.save('output/' + str(user_input[:12]) + "-negSample-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.png')

    return image_list, seed  # 返回图片列表和种子


def save_to_file(user_id, pos_prompt, neg_prompt, precision, special_requirements, common_sense, inspiration, aesthetics, abstract_understanding, seed, image_list, user_input=""):
    data = {
        "user_id": user_id,
        "positive_prompt": pos_prompt,
        "negative_prompt": neg_prompt,
        "description": user_input,
        "precision": precision,
        "special_requirements": special_requirements,
        "common_sense": common_sense,
        "inspiration": inspiration,
        "aesthetics": aesthetics,
        "abstract_understanding": abstract_understanding,
        "seed": seed  # 将种子保存到json文件中
    }
    
    with open('output/' + str(user_input[:12]) + "-" + str(user_id) + "-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.json', 'w', encoding="utf8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False)

    # 保存图片
    for img in image_list:
        # print(f'img : {img}')
        image_url = img['data']
        image_response = requests.get(image_url, stream=True)
        image_response.raise_for_status()  # 如果请求失败，这将引发一个异常
        with open('output/' + str(user_input[:12]) + "-" + str(user_id) + "-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.png', 'wb') as out_file:
            shutil.copyfileobj(image_response.raw, out_file)
        del image_response

def save_uploaded_image(image: Image, user_input: str):
    # 创建一个唯一的文件名，用用户输入的前12个字符和当前时间
    filename = f'output/{user_input[:12]}-upload-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.png'
    image.save(filename)


# 在 Gradio UI 中调用新的函数
with gr.Blocks(title="ChatGLM ArtAgent 数据标注") as demo:
    gr.Markdown("""<h1 align="center">🎊 ChatGLM ArtAgent 数据标注 🎊 </h1>""")
    seed = gr.State(0)  # 添加一个状态来保存种子

    # 添加一个并排的下拉菜单和说明栏
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=4):
                info = "我们的项目希望收集特定文字描述对应 prompt 的 golden answer。\n请您在右侧下拉栏中选择一个绘画描述，\n然后用英文填写效果最好的 positive prompt & negative prompt，\n图片生成后提交，并给出您的评分~"
                info_component = gr.Markdown(info)
            with gr.Column(scale=6):
                with open('prompt.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                options = [line.strip() for line in lines if line.strip() != '']  # 读取文件中的每一行作为下拉菜单的选项
                dropdown = gr.Dropdown(options, label="绘画描述")

    # Explanation section
    with gr.Box():
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():
                        positive_prompt = gr.Textbox(show_label=True, label="Positive Prompt", lines=4, value="((masterpiece, best quality, ultra-detailed, illustration)), ").style(container=True)
                        negative_prompt = gr.Textbox(show_label=True, label="Negative Prompt", lines=4, value="((nsfw: 1.2)), (EasyNegative:0.8), (badhandv4:0.8), (worst quality, low quality, extra digits), lowres, blurry, text, logo, artist name, watermarkhuman, 1girl, 1boy, loli, male, female, people, ").style(container=True)                    
                with gr.Column(scale=1):
                    gr.Markdown("生成图片后，请填写评价并提交，若再次点击此按钮将抛弃本次结果并重新生成")
                    drawBtn = gr.Button("Generate Image 🎨", variant="primary")
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Box(elem_id="对生成的图片进行评价"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                common_sense = gr.Slider(minimum=1, maximum=5, step=1, label="图片内元素是否【符合常识】", interactive=True)
                                aesthetics = gr.Slider(minimum=1, maximum=5, step=1, label="输出图像的【美学性】是否够强", interactive=True)
                                abstract_understanding = gr.Slider(minimum=1, maximum=5, step=1, label="输出图像是否可以【理解复杂和抽象的提示】，如情感和文化", interactive=True)
                                user_id = gr.Textbox(show_label=True, label="标注人学号（用于发放奖励）")
                            with gr.Column(scale=6):
                                precision = gr.Slider(minimum=1, maximum=5, step=1, label="【输出图像】与【下拉栏所选绘画描述 & prompt】是否一致，不遗漏元素", interactive=True)
                                special_requirements = gr.Slider(minimum=1, maximum=5, step=1, label="【输出图像】是否【符合】用户的【特殊要求】：构图方法、用光光线等", interactive=True)
                                inspiration = gr.Slider(minimum=1, maximum=5, step=1, label="是否能【启发绘画创作灵感】；若只给简单的prompt那么能不能【扩充意象】", interactive=True)
                                submitBtn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=4):
                    with gr.Tab("Gallery"):
                        result_gallery = gr.Gallery(label='Output', show_label=False, preview=True)
                    with gr.Tab("Upload"):
                        upload_image = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                        upload_image.change(save_uploaded_image, [upload_image, dropdown])
    
    history = gr.State([])
    result_list = gr.State([])

    drawBtn.click(call_sd_t2i, [positive_prompt, negative_prompt, dropdown], [result_gallery, seed], show_progress=True)
    submitBtn.click(save_to_file, [user_id, positive_prompt, negative_prompt, precision, special_requirements, common_sense, inspiration, aesthetics, abstract_understanding, seed, result_gallery, dropdown])
    upload_image.change(run_pnginfo, [upload_image, dropdown, user_id], outputs=[positive_prompt, negative_prompt])

demo.queue().launch(share=False, inbrowser=True, server_name='127.0.0.1', server_port=6006, favicon_path="./favicon.ico")