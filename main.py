import argparse
import os
import yaml
import time
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
from pathlib import Path
import tkinter as tk  
from tkinter import Canvas, Entry, Text, Button, PhotoImage, scrolledtext
import threading

# -------------------------- 获取当前代码文件所在目录（作为相对路径基准） --------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------- 全局变量与通用函数 --------------------------
is_detecting = False
detection_thread = None


# -------------------------- 第一个页面（首页）相关 --------------------------
def setup_first_page():
    # 1. 首页assets路径：改为相对路径（CURRENT_DIR/assets/frame0）
    ASSETS_PATH_FIRST = Path(os.path.join(CURRENT_DIR, "assets", "frame0"))

    def relative_to_assets_first(path: str) -> Path:
        return ASSETS_PATH_FIRST / Path(path)

    first_window = tk.Tk()
    first_window.geometry("1440x960")
    first_window.configure(bg="#FFFFFF")
    first_window.title("管内缺陷检测系统 - 首页")

    canvas_first = tk.Canvas(
        first_window,
        bg="#FFFFFF",
        height=960,
        width=1440,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas_first.place(x=0, y=0)

    canvas_first.create_text(
        484.0,
        98.0,
        anchor="nw",
        text="管内缺陷检测系统",
        fill="#000000",
        font=("Inter SemiBold", 48 * -1)
    )

    def jump_to_second_page():
        first_window.destroy()
        setup_second_page(model_path=args.model)

    try:
        button_image_enter = tk.PhotoImage(file=relative_to_assets_first("button_1.png"))
    except Exception as e:
        print(f"首页按钮图片加载失败（路径：{relative_to_assets_first('button_1.png')}）：{e}，将使用文本按钮")
        button_image_enter = None

    button_enter = tk.Button(
        first_window,
        image=button_image_enter if button_image_enter else None,
        text="进入系统" if not button_image_enter else "",
        borderwidth=0,
        highlightthickness=0,
        command=jump_to_second_page,
        relief="flat",
        font=("Inter SemiBold", 32 * -1) if not button_image_enter else None,
        fg="#000000" if not button_image_enter else None
    )
    button_enter.place(
        x=555.0,
        y=504.0,
        width=360.0,
        height=109.0
    )

    if button_image_enter:
        button_enter.image = button_image_enter

    first_window.resizable(False, False)
    first_window.mainloop()


# -------------------------- 第二个页面（检测页）相关 --------------------------
def draw_bounding_box(img, class_id, confidence, x1, y1, x2, y2, colors, classes, terminal_widget):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    label = f"{classes[class_id]} ({confidence:.2f})"
    color = tuple(map(int, colors[class_id]))

    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 14)
    except Exception as e:
        font = ImageFont.load_default()
        update_terminal(f"警告：中文字体加载失败（路径：/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc）：{e}，中文标签可能无法显示", terminal_widget)

    text_x = max(x1 - 10, 10)
    text_y = max(y1 - 25, 10)
    text_bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle(text_bbox, fill=(0, 0, 0))
    draw.text((text_x, text_y), label, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def update_terminal(text, terminal_widget):
    terminal_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
    terminal_widget.see(tk.END)


def detection_loop(model_path, camera_id, save_dir, save_interval, video_label, terminal_widget):
    global is_detecting
    os.makedirs(save_dir, exist_ok=True)
    update_terminal(f"检测结果保存目录（完整路径）：{os.path.abspath(save_dir)}", terminal_widget)

    try:
        # data.yaml路径：默认和2.py同目录（CURRENT_DIR/data.yaml），无需修改（check_yaml会自动在当前目录找）
        yaml_path = check_yaml("data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"data.yaml文件未找到（当前路径：{os.getcwd()}），请放在脚本同目录")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            classes = yaml.safe_load(f)["names"]
        if isinstance(classes, list):
            classes = {i: cls for i, cls in enumerate(classes)}
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        update_terminal(f"成功加载类别文件（路径：{yaml_path}），类别：{list(classes.values())}", terminal_widget)
    except Exception as e:
        update_terminal(f"加载类别文件失败：{str(e)}", terminal_widget)
        return

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到（完整路径：{model_path}）")
        model = YOLO(model_path, task='detect')
        update_terminal(f"成功加载模型（完整路径：{model_path}）", terminal_widget)
    except Exception as e:
        update_terminal(f"加载模型失败：{str(e)}", terminal_widget)
        return

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened():
        update_terminal(f"无法打开摄像头（ID：{camera_id}），请检查摄像头连接或更换ID（如1）", terminal_widget)
        return

    last_save_time = 0
    try:
        while is_detecting:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                update_terminal("无法获取视频帧，重试中...", terminal_widget)
                time.sleep(0.5)
                continue

            results = model(frame, conf=0.1, iou=0.4, verbose=False)
            detected = False
            for result in results:
                for box in result.boxes:
                    detected = True
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                    frame = draw_bounding_box(frame, class_id, confidence, x1, y1, x2, y2, colors, classes, terminal_widget)
                    update_terminal(f"检测到：{classes[class_id]}（置信度：{confidence:.2f}，位置：({x1},{y1})-({x2},{y2})）", terminal_widget)

            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            video_label.imgtk = img_tk
            video_label.config(image=img_tk)

            current_time = time.time()
            if detected and (current_time - last_save_time) > save_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"detection_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                update_terminal(f"已保存检测结果（完整路径）：{save_path}", terminal_widget)
                last_save_time = current_time

            time.sleep(0.01)
    finally:
        cap.release()
        update_terminal("检测已停止，摄像头资源已释放", terminal_widget)


def start_detection(video_label, terminal_widget, model_path):
    global is_detecting, detection_thread
    if not is_detecting:
        is_detecting = True
        detection_thread = threading.Thread(
            target=detection_loop,
            args=(model_path, 0, "detection_results", 3, video_label, terminal_widget),
            daemon=True
        )
        detection_thread.start()
        update_terminal("开始实时检测（点击「停止检测」按钮终止）", terminal_widget)
    else:
        update_terminal("检测已在运行中，无需重复启动", terminal_widget)


def stop_detection(terminal_widget):
    global is_detecting
    if is_detecting:
        is_detecting = False
        update_terminal("正在停止检测...（等待当前帧处理完成）", terminal_widget)
    else:
        update_terminal("检测未运行，无需停止", terminal_widget)


def exit_program(window):
    global is_detecting
    is_detecting = False
    time.sleep(0.5)
    window.destroy()


def setup_second_page(model_path):

    ASSETS_PATH_SECOND = Path(os.path.join(CURRENT_DIR, "assets", "frame1"))

    def relative_to_assets_second(path: str) -> Path:
        return ASSETS_PATH_SECOND / Path(path)

    second_window = tk.Tk()
    second_window.geometry("1440x960")
    second_window.configure(bg="#FFFFFF")
    second_window.title("管内缺陷检测系统 - 实时检测")

    canvas_second = tk.Canvas(
        second_window,
        bg="#FFFFFF",
        height=960,
        width=1440,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas_second.place(x=0, y=0)

    canvas_second.create_rectangle(684.0, 161.0, 1332.0, 641.0, fill="#D9D9D9", outline="")
    video_label = tk.Label(second_window, bg="#D9D9D9")
    video_label.place(x=688.0, y=165.0, width=640, height=480)

    canvas_second.create_rectangle(106.0, 161.0, 466.0, 641.0, fill="#D9D9D9", outline="")
    terminal_widget = scrolledtext.ScrolledText(
        second_window,
        wrap=tk.WORD,
        bg="#333333",
        fg="#FFFFFF",
        font=("Consolas", 10)
    )
    terminal_widget.place(x=110.0, y=165.0, width=452, height=472)

    # 3.1 开始检测按钮
    try:
        btn1_img = tk.PhotoImage(file=relative_to_assets_second("button_1.png"))
    except Exception as e:
        print(f"检测页按钮1图片加载失败（路径：{relative_to_assets_second('button_1.png')}）：{e}，使用文本按钮")
        btn1_img = None
    btn_start = tk.Button(
        second_window,
        image=btn1_img if btn1_img else None,
        text="开始检测" if not btn1_img else "",
        borderwidth=0,
        highlightthickness=0,
        command=lambda: start_detection(video_label, terminal_widget, model_path),
        relief="flat"
    )
    btn_start.place(x=777.0, y=768.0, width=180.0, height=80.0)
    if btn1_img:
        btn_start.image = btn1_img

    # 3.2 停止检测按钮
    try:
        btn2_img = tk.PhotoImage(file=relative_to_assets_second("button_2.png"))
    except Exception as e:
        print(f"检测页按钮2图片加载失败（路径：{relative_to_assets_second('button_2.png')}）：{e}，使用文本按钮")
        btn2_img = None
    btn_stop = tk.Button(
        second_window,
        image=btn2_img if btn2_img else None,
        text="停止检测" if not btn2_img else "",
        borderwidth=0,
        highlightthickness=0,
        command=lambda: stop_detection(terminal_widget),
        relief="flat"
    )
    btn_stop.place(x=1091.0, y=768.0, width=180.0, height=80.0)
    if btn2_img:
        btn_stop.image = btn2_img

    # 3.3 退出系统按钮
    try:
        btn3_img = tk.PhotoImage(file=relative_to_assets_second("button_3.png"))
    except Exception as e:
        print(f"检测页按钮3图片加载失败（路径：{relative_to_assets_second('button_3.png')}）：{e}，使用文本按钮")
        btn3_img = None
    btn_exit = tk.Button(
        second_window,
        image=btn3_img if btn3_img else None,
        text="退出系统" if not btn3_img else "",
        borderwidth=0,
        highlightthickness=0,
        command=lambda: exit_program(second_window),
        relief="flat"
    )
    btn_exit.place(x=196.0, y=768.0, width=180.0, height=80.0)
    if btn3_img:
        btn_exit.image = btn3_img

    # 检测页窗口的主循环
    second_window.resizable(False, False)
    second_window.mainloop()


# -------------------------- 程序入口 ----------------
if __name__ == "__main__":
    # 解析命令行参数：用于指定YOLO模型路径（以下3行都在if __name__代码块内，缩进一致）
    parser = argparse.ArgumentParser(description="管内缺陷检测系统")
    parser.add_argument(
        "--model",  # 命令行参数名（运行时可通过 --model 手动指定模型路径）
        default=os.path.join(CURRENT_DIR, "best.onnx"),  # 默认模型路径
        help="YOLO模型文件路径（支持 .onnx, .pt 等格式，例：--model ./my_model.pt）"
    )

    args = parser.parse_args()

    setup_first_page()
