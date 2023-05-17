# ChatGLM-ArtAgent

## 介绍

* P-tuned ChatGLM-6B -> AutoGPT -> LDM

### 安装指南

**注意：本项目使用了 ChatGLM-6B 和 Stable Diffusion 这意味着，您需要有足够的算力来同时运行这两个模型。本项目本身没有 GPU 开销，因此，推荐您将 ChatGLM-6B 与 Stable Diffusion部署在您的计算型服务器上**

1. clone 本项目并安装依赖

```shell
$ git clone https://github.com/tuteng0915/ChatGLM-ArtAgent.git
$ cd ChatGLM-ArtAgent
$ pip install -r requirements.txt
```

2. 安装 nltk 模型及数据

方法一：通过 python nltk 库安装

```shell
$ python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
```

方法二：通过此链接下载数据：`https://cloud.tsinghua.edu.cn/f/1831442f5e734d7da61a/`
```shell
$ python
>>> import nltk
>>> nltk.data.path
```
将文件解压后，置于该命令所打印的任一位置下即可


3. 调整 ChatGLM-6B 和 Stable Diffusion 接口
```python
# ./utils.py/call_glm_api
def call_glm_api(prompt, history, max_length, top_p, temperature):
    url = "http://127.0.0.1:8000"       # 将改行修改为您部署的 ChatGLM-6B api 地址

# ./utils.py/call_sd_t2i
def call_sd_t2i(pos_prompt, neg_prompt, width, height, steps, user_input=""):
    url = "http://127.0.0.1:6016"       # 将该行修改为 AUTOMATIC1111/stable-diffusion-webui 地址，并开放api
```

<!-- 4. 下载 promptgen_model 置于 ./model/promptgen-lexart

https://cloud.tsinghua.edu.cn/d/e2797260a8f94ba994dd/ -->

4. 运行
```shell
$ python art_agent.py
# 可以看到如下输出:
promptgen_model loaded
danbooru tags loaded
Running on local URL:  http://127.0.0.1:6006
```
浏览器访问 `localhost:6006` 即可。

注意，如果您可以成功打开网页

#### 硬件需求

#### 环境安装

#### 接口调用

### 使用指南



## TODO LIST
* 1 完善 Readme, About us, UI 等
    * 1.1 Readme 中的介绍
    * 1.2 Readme 中的安装指南
        * requirements.txt 可能不全
    * 1.3 Readme 中的使用指南
        * 包括各类 case
    * 1.4 About us
        * 更新在 Readme 和 art_agent.py 中
    * 1.5 更多 UI 细节
    * 1.6 Readme-ENG

* 2 建立评测体系：
    * 2.1 构建评测 prompt 集
        * 更新在 ./benchmark/prompts.txt 中，应尽可能考虑到各种类型的 prompt
    * 2.2 构建评价指标
        * 不仅仅考虑一致性，也需要考虑多样性，更新在 ./benckmark 中
    * 2.3 自动评价、汇报
        * 构建自动评测脚本，脚本需要方便直观地给出链路在各种优化中对于 benchmark/prompts.txt 的改变

* 3 利用构建艺术评论领域的问答数据集并 finetune GLM-6B 模型 (WC) 
    * 3.1 跑通 GLM-6B 的 finetune
        * 参考 https://github.com/THUDM/ChatGLM-6B
    * 3.2 进行调研，确定数据集来源
        * 哪个领域，哪些书，哪些章节
    * 3.3 使用 GPT-4 作为辅助，构建问答数据集
        * 给出书的下文，让 GPT 预测提问
    * 3.4 对 GLM-6B 进行 finetune 并评价其效果
        * 
* 4 多种方式优化 Natural Language -> Prompt 模块 (FSJ)
    * 4.1 更长的问答链路
        * 先介绍 -> 再概况 -> 给出画面 -> 总结元素
    * 4.2 补充 Prompts 的专用模型
        * Seq2Seq? 需要调研和尝试
    * 4.3 匹配关键词 
    * 4.4 补充固定关键词

* 5 实现类 autoGPT 链路，并做出创新性优化 (WLJ)
    * 5.1 调研 autoGPT
        * 分析优势劣势, 如何实现
    * 5.2 实现 autoGPT

* 6 图像理解
    * 6.1 上传图像

* 7 丰富功能
    * 7.1 显示当前 prompts
    * 7.2 直接与 SD 交互，类似于 Midjourney

* 8 


## Contact us!

* Email: 
    * leotuteng@126.com
* WeChat:
    * TT-20000915-tuteng


## 参考

* Forked from https://github.com/THUDM/ChatGLM-6B

* https://github.com/AUTOMATIC1111/stable-diffusion-webui

* https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen

* https://github.com/LemonQu-GIT/ChatGLM-6B-Engineering

* https://github.com/DominikDoom/a1111-sd-webui-tagcomplete


## 报错答疑

* 形如这样的错误，一般来源于 api 配置问题。请检查 ChatGLM-6B api 是否已部署其端口配置正确，检查 AUTOMATIC1111/stable-diffusion-webui 是否已部署且开启了 --api
```shell
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: / (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021A087AFD30>: Failed to establish a new connection: [WinError 10061] 由于目标计算机积极 拒绝，无法连接。'))
```
