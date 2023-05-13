import os
from typing import Dict, Tuple, Union, Optional
from torch.nn import Module
from transformers import AutoModel
import requests, re, json, io, base64, os
from urllib.parse import quote
import mdtex2html
from bs4 import BeautifulSoup
from PIL import Image, PngImagePlugin
from transformers import AutoModel, AutoTokenizer
import gradio as gr
from promptgen import *
import time
import random
import re
from tqdm import tqdm


# glm_tokenizer = AutoTokenizer.from_pretrained("./model/ChatGLM-6B", trust_remote_code=True)
glm_tokenizer = None
# glm_model = AutoModel.from_pretrained("./model/ChatGLM-6B", trust_remote_code=True).half().quantize(4).cuda()
glm_model = None
# glm_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# glm_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# glm_model = glm_model.eval()

def predict(input, chatbot, max_length, top_p, temperature, history, from_api=True):
    chatbot.append((parse_text(input), ""))
    if not from_api:
        for response, history in glm_model.stream_chat(glm_tokenizer, input, history, max_length=max_length, top_p=top_p,
                                                temperature=temperature):
            chatbot[-1] = (parse_text(input), parse_text(response))       
            yield chatbot, history
    else:
        response = call_glm_api(input, history, max_length, top_p, temperature)["response"]
        chatbot[-1] = (parse_text(input), parse_text(response)) 
        history.append([chatbot[-1][0], chatbot[-1][1]])
        yield chatbot, history
    print(history)


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

def clear_gallery():
    return [], []


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def translate(word):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    response = requests.post(url, data=key)
    if response.status_code == 200:
        list_trans = response.text
        result = json.loads(list_trans)
        trans = ""
        for r in result['translateResult'][0]:
            trans += r['tgt']
        return trans
    else:
        print(response.status_code)
        return word


def call_sd_t2i(pos_prompt, neg_prompt, width, height, steps, cfg, user_input=""):
    url = "http://127.0.0.1:6016"
    payload = {
        "enable_hr": True,
        "denoising_strength": 0.55,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "prompt": pos_prompt,
        "steps": steps,
        "negative_prompt": neg_prompt,
        "cfg_scale": cfg,
        "batch_size": 1,
        "n_iter": 1,
        "width": width,
        "height": height,
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    # print(r)
    image_list = []
    os.makedirs('output', exist_ok=True)
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        
        # Get Image Info
        # png_payload = {"image": "data:image/png;base64," + i}        
        # response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
        # pnginfo = PngImagePlugin.PngInfo()
        # pnginfo.add_text("parameters", response2.json().get("info"))
        image.save('output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input[:12]) + "-" + str(random.randint(1000, 9999)) +'.png')

    return image_list


def call_glm_api(prompt, history, max_length, top_p, temperature):
    url = "http://127.0.0.1:8000"
    payload = {
        "prompt": prompt,
        "history": history,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature
    }
    response = requests.post(url, json=payload)
    response = response.json()
    # print(response)
    return response


def gen_image_description(user_input, chatbot, max_length, top_p, temperature, history, file_handle, from_api=True):
    # TODO 4.1
    def get_respond(prompt_history, prompt_input):
        if not from_api:
            response = ""
            for response_, history_ in glm_model.stream_chat(glm_tokenizer, prompt_input, prompt_history, max_length=max_length, top_p=top_p,
                                                temperature=temperature):
                response = response_
        else:
            response = call_glm_api(prompt_input, prompt_history, max_length, top_p, temperature)["response"]
        return response
    def write_log():
        file_handle.write("="*20 + "\n")
        file_handle.write("prompt_history:" + str(prompt_history) + "\n")
        file_handle.write("prompt_input:" + prompt_input + "\n")
        file_handle.write("response:" + response + "\n\n")
        file_handle.write("="*20+ "\n")

    # Step1 名词解释
    prompt_history = [["我接下来会给你一些名词，请你依次给出它们的解释。","好的，请给我一些指令。"]]
    prompt_input = str(f"名词解释：“{user_input}”，请详细解释这些词，并添加一些形象和内容以丰富细节，不要输出多余的信息")
    response = get_respond(prompt_history, prompt_input)
    print("Step1", response)
    write_log()

    # Step2 元素提取和总结
    prompt_history.append([prompt_input, response])
    prompt_input = str(f"请总结归纳你刚刚的解释，并为其添加一些视觉上的元素和细节，不要输出多余的信息。")
    response = get_respond(prompt_history, prompt_input)
    print("Step2", response)
    write_log()

    # Step3 作画素材
    prompt_history = [["我接下来会给你一些作画的指令，你只要回复出作画内容及对象，不需要你作画，不需要给我参考，请直接给出作画内容，不要输出不必要的内容，你只需回复作画内容。你听懂了吗", "听懂了。请给我一些作画的指令。"]]
    prompt_input = str(f"我现在要画一副画，这幅画关于：{response}。请帮我详细描述作画中的画面构图、画面主体和画面背景，并添加一些内容以丰富细节。回答中不要包含这一句话")
    response = get_respond(prompt_history, prompt_input)
    print("Step3", response)
    write_log()

    # 检测
    # retry_count = 0
    # check = get_respond([], str(f"这里有一段描述，{response}，这段描述是关于一个场景的吗？你仅需要回答“是”或“否”。"))
    # print("CHECK", check)
    # while ("不是" in check or "是" not in check) and retry_count < 3:
    #     response = get_respond(prompt_history, prompt_input)
    #     check = get_respond([], str(f"这里有一段描述，{response}，这段描述是关于一个场景、物体、动物或人物的吗？你仅需要回答“是”或“否”。"))
    #     retry_count += 1

    # if "不是" in check or "是" not in check:
    #     response = "抱歉, 我还不知道该怎么画，我可能需要更多学习。"
    #     chatbot.append((parse_text(user_input), parse_text(response)))
    #     history.append([chatbot[-1][0], chatbot[-1][1]])
    #     return chatbot, history, parse_text(response), "FAILED"

    chatbot.append((parse_text("请帮我画："+user_input), parse_text(response)))
    history.append([chatbot[-1][0], chatbot[-1][1]])

    # Step4 作画素材
    prompt_history = [["下面我将给你一段话，请你帮我抽取其中的图像元素，忽略其他非图像的描述，将抽取结果以逗号分隔，一定不要输出多余的内容和符号","听懂了，请给我一段文字。"]]
    prompt_input = str(f"以下是一段描述，抽取其中包括{TAG_STRING}的图像元素，忽略其他非图像的描述，将抽取结果以逗号分隔：{response}。 {user_input}")
    response = get_respond(prompt_history, prompt_input)
    print("Step4", response)
    write_log()

    # print(history[-1])
    return chatbot, history, parse_text(response) + "\n" + chatbot[-1][1], "SUCCESS"



def sd_predict(user_input, chatbot, max_length, top_p, temperature, history, width, height, steps, cfg, result_list):
    file_handle = open('output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input[:12]) + '.txt', 'w', encoding="utf8")
    
    # Step 1 use ChatGLM-6B associate image description
    # !!! Currently, we don't take history into consideration
    chatbot, history, image_description, code = gen_image_description(user_input, chatbot, max_length, top_p, temperature, history, file_handle)

    if code != "SUCCESS":
        yield chatbot, history, result_list, result_list
    else:
        # image_description = history[-1][1]
        # image_description = str("").join(image_description.split('\n')[1:])
        # stop_words = ["好的", "我", "将", "会", "画作", "关于", "一张", "画"]
        stop_words = ["\t", "\r", '-', '*', '·', "<br>"]
        for word in stop_words:
            image_description = image_description.replace(word, "\n") + "\n"
        # print(image_description)
        tag_dict = {}

        # base on re        
        # for tag_class in TAG_CLASSES + ["构图", "主体", "背景", "内容"]:
        #     pat = r'{}：.*[\n]'.format(tag_class)
        #     # print(pat)
        #     pat = re.compile( pat)
        #     find = pat.findall(image_description)
        #     if len(find) > 0:
        #         if "不清楚" not in find[0] and "无" not in find[0] and "没有描述" not in find[0] and "不知道" not in find[0] and len(find[0]) > 1:
        #             tag_dict[tag_class] = find[0][len(tag_class) + 1: -1]
        
        # base on find
        TAG_CLASSES_ = TAG_CLASSES + SIMI_TAG_CLASSES
        tag_pos_dict = {}
        for t in TAG_CLASSES_:
            pos = image_description.find(t+"：")
            if pos != -1:
                tag_pos_dict[t] = pos
        tag_pos_dict = sorted(tag_pos_dict.items(), key = lambda kv:(kv[1], kv[0]))
        tag_pos_dict = [(index, a[0], a[1]) for index, a in enumerate(tag_pos_dict)] + [(len(tag_pos_dict), "", len(image_description))]
        print(tag_pos_dict)
        for index in range(len(tag_pos_dict) - 1):
            l = tag_pos_dict[index][2] + len(tag_pos_dict[index][1]) + 1
            r = tag_pos_dict[index+1][2]
            tmp = image_description[l:r]
            if "不清楚" not in tmp and "无" not in tmp and "没有描述" not in tmp and "不知道" not in tmp and "未指定" not in tmp:
                tmp = tmp.replace('\n', "")
                tag_dict[tag_pos_dict[index][1]] = tmp

        print(tag_dict)
        file_handle.write(str(tag_dict) + "\n")
        
        if len(tag_dict) <= 1:
            for word in TAG_CLASSES + ["\n", "\t", "\r", "<br>"] + ["根据描述无法识别", "无", "没有描述", "不知道", "不清楚"]:
                image_description = image_description.replace(word, ", ")
            tag_dict["其他"] = image_description
            print(tag_dict)
        
        tag_dict = dict([(tag, translate(tag_dict[tag]).lower()) for tag in tag_dict if len(tag_dict[tag]) > 0])
        print(tag_dict)        
        file_handle.write(str(tag_dict) + "\n")
        # image_description = translate(image_description)
        # print(image_description)

        # Step 2 use promprGenerater get Prompts
        # prompt_list = gen_prompts(image_description, batch_size=4)
        # print(prompt_list)
        # yield chatbot, history, result_list, []

        # Alternative plan
        # prompt_list = [ enhance_prompts(image_description) ] * 4
        prompt_list = tag_extract(tag_dict)
        print(prompt_list[0])
        file_handle.write("\n".join(["Prompt:"+p[0]+"\nNegative Prompt:"+p[1] for p in prompt_list]))

        # Show Prompts
        prompt_text = "\n\n Prompt:\n\n " + str(prompt_list[0][0]) + "\n\nNegative Prompt: \n\n" + str(prompt_list[0][1])
        chatbot[-1] = (chatbot[-1][0], chatbot[-1][1] + prompt_text)


        file_handle.close()

        # Step 3 use SD get images
        for pos_prompt, neg_prompt in tqdm(prompt_list):
            new_images = call_sd_t2i(pos_prompt, neg_prompt, width, height, steps, cfg, user_input)
            result_list = result_list + new_images
            yield chatbot, history, result_list, new_images
        yield chatbot, history, result_list, result_list

