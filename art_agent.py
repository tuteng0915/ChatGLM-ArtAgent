import gradio as gr
from utils import *

greetings = [("你好呀！", "您好！我是 ChatGLM-ArtAgent，一个与您交流艺术构思的AI助手。 \n\n 我调用了 ChatGLM-6B LLM 模型，和 Stable Diffusion LDM 模型。\n\n 我目前只擅长生成景物，偶尔会不受控地生成人物。 \n\n 我还在测试阶段，链路中也存在很多随机性： \n\n Sometimes a simple retry can make it betteer.")]

gr.Chatbot.postprocess = postprocess

with gr.Blocks(title="ChatGLM ArtAgent") as demo:
    gr.HTML("""<h1 align="center">🎊 ChatGLM ArtAgent 🎊 </h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(greetings).style(height=640)
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(
                            container=False)
                    with gr.Column(scale=1, min_width=100):
                        submitBtn = gr.Button("Chat with GLM 🚀",)
                        drawBtn = gr.Button("Generate Image 🎨", variant="primary")
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Tab("Gallery"):
                    result_gallery = gr.Gallery(label='Output', show_label=False).style(preview=True)
                with gr.Tab("Upload Image"):
                    # TODO 6.1
                    upload_image = gr.Image(label='Upload', show_label=True)
            with gr.Row():
                with gr.Tab("Settings"):
                    with gr.Tab(label="Stable Diffusion"):
                        with gr.Column(min_width=100):
                            # clearBtn = gr.Button("Clear Gallery")
                            with gr.Row():
                                sd_width = gr.Slider(512, 1024, value=768, step=32, label="Width", interactive=True)
                                sd_height = gr.Slider(512, 1024, value=768, step=32, label="Height ", interactive=True)
                            with gr.Row():
                                sd_steps = gr.Slider(8, 40, value=32, step=4, label="Steps", interactive=True)
                                sd_cfg = gr.Slider(4, 20, value=7, step=0.5, label="CFG Scale", interactive=True)
                            with gr.Row():
                                sd_batch_num = gr.Slider(1, 8, value=4, step=1, label="Batch Num", interactive=True)
                                sd_batch_size = gr.Slider(1, 8, value=2, step=1, label="Batch Size", interactive=True)
                    with gr.Tab(label="ChatGLM-6B"):
                        with gr.Column(min_width=100):
                            # emptyBtn = gr.Button("Clear History")
                            max_length = gr.Slider(0, 4096, value=2048, step=64.0, label="Maximum length", interactive=True)
                            with gr.Row():
                                top_p = gr.Slider(0, 1, value=0.6, step=0.01, label="Top P", interactive=True)
                                temperature = gr.Slider(0, 1, value=0.90, step=0.01, label="Temperature", interactive=True)
                            # TODO 5.2
                            self_chat_round = gr.Slider(0, 3, value=0, step=0, label="Under Development", interactive=True)  # Self Chat Round
                            prompt_mask_ratio = gr.Slider(0, 1, value=0.8, step=0.05, label="Under Development", interactive=True)
                with gr.Tab("More Actions"):
                    # TODO 7.1
                    c1 = gr.HTML("under development")
                with gr.Tab("About Us"):
                    # TODO 1.4
                    c2 = gr.HTML("under development")

    history = gr.State([])
    result_list = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    drawBtn.click(sd_predict, [user_input, chatbot, max_length, top_p, temperature, history, sd_width, sd_height, sd_steps, sd_cfg, result_list],
                     [chatbot, history, result_list, result_gallery], show_progress=True)
    drawBtn.click(reset_user_input, [], [user_input])

    # emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    # clearBtn.click(clear_gallery, outputs=[result_list, result_gallery], show_progress=True)

os.makedirs('output', exist_ok=True)
demo.queue().launch(share=False, inbrowser=True, server_name='127.0.0.1', server_port=6006, favicon_path="./favicon.ico")
