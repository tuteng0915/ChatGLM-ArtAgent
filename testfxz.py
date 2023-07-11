from utils import sd_predict

def main():
    with open('prompt.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for user_input in lines:
        user_input = user_input.strip()  # remove newline characters
        if user_input:  # if the line is not empty
            chatbot = []
            max_length = 5000
            top_p = 0.9
            temperature = 0.8
            history = []
            width = 512
            height = 512
            steps = 500
            cfg = 1.0
            result_list = []

            for chatbot, history, result_list, new_images in sd_predict(user_input, chatbot, max_length, top_p, temperature, history, width, height, steps, cfg, result_list):
                if isinstance(new_images, (list, tuple)):  # check if new_images is iterable
                    for image in new_images:
                        print("one epoch")

if __name__ == "__main__":
    main()

# 批量生成图片测试集的程序

# prompt.txt的前16条是easy cases，后16条是hard cases
# 人物/景物、长/短prompt、超自然（比如狮子当老师）
# 只允许变动positive prompt，使用同一个绘画模型