import cv2
import os
import time

# 设置保存路径
SAVE_DIR = "D:/yolov8-2.0-train-image-video/ultralytics/LSTM+Tracking/data/cam_test/front-back/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置摄像头索引 (0 为默认摄像头)
cap = cv2.VideoCapture(1)

# 视频参数
fps = 30  # 采样帧率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

video_count = 0  # 记录采集视频数量
recording = False  # 是否正在录制
video_writer = None

print("按 'r' 开始/停止录制, 按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 显示实时画面
    cv2.imshow("Camera", frame)

    # 如果正在录制，写入视频
    if recording and video_writer is not None:
        video_writer.write(frame)

    # 监听按键
    key = cv2.waitKey(1) & 0xFF

    # 按 'r' 开始/停止录制
    if key == ord('r'):
        if not recording:
            # 生成视频文件名
            video_count += 1
            video_name = os.path.join(SAVE_DIR, f"video_{video_count}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))
            recording = True
            print(f"开始录制: {video_name}")
        else:
            recording = False
            video_writer.release()
            print(f"录制完成: video_{video_count}.mp4")

    # 按 'q' 退出
    elif key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
