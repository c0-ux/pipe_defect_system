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
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, scrolledtext
import threading

# 全局变量控制检测状态
is_detecting = False
detection_thread = None

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"E:\raspberrrypi\build\assets\frame1")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def draw_bounding_box(img, class_id, confidence, x1, y1, x2, y2, colors, classes):
    """绘制边界框和标签（支持中文显示）"""
    # 将OpenCV图像（BGR格式）转换为PIL图像（RGB格式）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 准备标签文本
    label = f"{classes[class_id]} ({confidence:.2f})"
    color = tuple(map(int, colors[class_id]))  # 将numpy数组转换为PIL可用的元组

    # 绘制边界框
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    # 加载中文字体
    try:
        # 树莓派系统
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 14)
    except:
        # Windows系统备选
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 14)
        except:
            # 如果找不到中文字体，使用默认字体
            font = ImageFont.load_default()
            print("警告：未找到中文字体，可能无法正常显示中文标签")

    # 计算文本位置
    text_x = max(x1 - 10, 10)
    text_y = max(y1 - 25, 10)

    # 绘制文本背景
    text_bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle(text_bbox, fill=(0, 0, 0))
    # 绘制文本
    draw.text((text_x, text_y), label, font=font, fill=color)

    # 将PIL图像转回OpenCV格式（BGR）
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def update_terminal(text, terminal_widget):
    """更新终端显示"""
    terminal_widget.insert(tk.END, text + "\n")
    terminal_widget.see(tk.END)  # 自动滚动到底部


def detection_loop(model_path, camera_id, save_dir, save_interval, video_label, terminal_widget):
    """目标检测循环，在单独线程中运行"""
    global is_detecting

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    update_terminal(f"检测结果将保存至: {os.path.abspath(save_dir)}", terminal_widget)

    # 加载类别信息
    try:
        yaml_path = check_yaml("data.yaml")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            classes = yaml.safe_load(f)["names"]
        # 如果类别是列表格式，转换为字典
        if isinstance(classes, list):
            classes = {i: cls for i, cls in enumerate(classes)}
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
    except Exception as e:
        update_terminal(f"加载类别加载类别文件失败: {str(e)}", terminal_widget)
        return

    # 加载ONNX模型
    try:
        model = YOLO(model_path, task='detect')
        update_terminal(f"成功加载模型: {model_path}", terminal_widget)
    except Exception as e:
        update_terminal(f"加载ONNX模型失败: {str(e)}", terminal_widget)
        return

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        update_terminal("无法打开摄像头", terminal_widget)
        return

    # 初始化保存计时器
    last_save_time = 0

    try:
        while is_detecting:
            start_time = time.time()
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                update_terminal("无法获取视频帧，重试中...", terminal_widget)
                time.sleep(0.5)
                continue

            # 执行推理
            results = model(frame, conf=0.2, iou=0.4, verbose=False)

            # 处理检测结果
            detected = False
            for result in results:
                for box in result.boxes:
                    detected = True
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, frame.shape[1]))
                    y1 = max(0, min(y1, frame.shape[0]))
                    x2 = max(0, min(x2, frame.shape[1]))
                    y2 = max(0, min(y2, frame.shape[0]))

                    # 绘制边界框和中文标签
                    frame = draw_bounding_box(
                        frame, class_id, confidence, x1, y1, x2, y2, colors, classes
                    )

                    # 在终端显示检测信息
                    update_terminal(
                        f"{datetime.now().strftime('%H:%M:%S')} 检测到: {classes[class_id]} "
                        f"置信度: {confidence:.2f}", terminal_widget
                    )

            # 计算并显示帧率
            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 转换图像格式并显示在Tkinter窗口
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # 更新视频标签
            video_label.imgtk = img_tk
            video_label.config(image=img_tk)

            # 保存检测结果（有目标且达到保存间隔）
            current_time = time.time()
            if detected and (current_time - last_save_time) > save_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"detection_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                update_terminal(f"已保存检测结果: {save_path}", terminal_widget)
                last_save_time = current_time

            # 短暂延迟，降低CPU占用
            time.sleep(0.01)

    finally:
        # 释放资源
        cap.release()
        update_terminal("检测已停止", terminal_widget)


def start_detection(video_label, terminal_widget, model_path="yolov8n.onnx"):
    """开始检测按钮回调"""
    global is_detecting, detection_thread

    if not is_detecting:
        is_detecting = True
        # 在新线程中启动检测，避免界面卡顿
        detection_thread = threading.Thread(
            target=detection_loop,
            args=(model_path, 0, "detection_results", 3, video_label, terminal_widget),
            daemon=True
        )
        detection_thread.start()
        update_terminal("开始实时检测...", terminal_widget)


def stop_detection(terminal_widget):
    """停止检测按钮回调"""
    global is_detecting
    is_detecting = False
    update_terminal("正在停止检测...", terminal_widget)


def exit_program(window):
    """退出程序按钮回调"""
    global is_detecting
    is_detecting = False
    # 给线程一点时间停止
    time.sleep(0.5)
    window.destroy()


def setup_gui():
    """设置GUI界面"""
    window = Tk()
    window.geometry("1440x960")
    window.configure(bg="#FFFFFF")
    window.title("目标检测系统")

    canvas = Canvas(
        window,
        bg="#FFFFFF",
        height=960,
        width=1440,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.place(x=0, y=0)

    # 创建视频显示区域（640x480）
    canvas.create_rectangle(
        684.0,
        161.0,
        1332.0,  # 684 + 648 (稍微大一点的区域)
        641.0,  # 161 + 480
        fill="#D9D9D9",
        outline=""
    )

    # 创建视频标签
    video_label = tk.Label(window, bg="#D9D9D9")
    video_label.place(x=688.0, y=165.0, width=640, height=480)

    # 创建终端显示区域
    canvas.create_rectangle(
        106.0,
        161.0,
        466.0,
        641.0,
        fill="#D9D9D9",
        outline=""
    )

    # 创建带滚动条的文本框作为终端
    terminal = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        bg="#333333",
        fg="#FFFFFF",
        font=("SimHei", 10)
    )
    terminal.place(x=110.0, y=165.0, width=452, height=472)

    # 开始检测按钮
    try:
        button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    except:
        # 如果图片加载失败，使用文本按钮替代
        button_image_1 = None

    button_1 = Button(
        image=button_image_1 if button_image_1 else None,
        text="开始检测" if not button_image_1 else "",
        borderwidth=0,
        highlightthickness=0,
        command=lambda: start_detection(video_label, terminal),
        relief="flat"
    )
    button_1.place(
        x=777.0,
        y=768.0,
        width=180.0,
        height=80.0
    )

    # 停止检测按钮
    try:
        button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
    except:
        button_image_2 = None

    button_2 = Button(
        image=button_image_2 if button_image_2 else None,
        text="停止检测" if not button_image_2 else "",
        borderwidth=0,
        highlightthickness=0,
        command=lambda: stop_detection(terminal),
        relief="flat"
    )
    button_2.place(
        x=1091.0,
        y=768.0,
        width=180.0,
        height=80.0
    )

    # 退出系统按钮
    try:
        button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
    except:
        button_image_3 = None

    button_3 = Button(
        image=button_image_3 if button_image_3 else None,
        text="退出系统" if not button_image_3 else "",
        borderwidth=0,
        highlightthickness=0,
        command=lambda: exit_program(window),
        relief="flat"
    )
    button_3.place(
        x=196.0,
        y=768.0,
        width=180.0,
        height=80.0
    )

    window.resizable(False, False)
    return window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="带GUI的YOLO目标检测程序")
    parser.add_argument("--model", default="yolov8n.onnx", help="ONNX模型路径（默认：yolov8n.onnx）")
    args = parser.parse_args()

    # 设置中文字体支持
    root = setup_gui()
    root.mainloop()
