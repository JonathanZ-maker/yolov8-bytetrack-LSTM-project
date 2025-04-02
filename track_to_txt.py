import numpy as np
import os
import cv2
from yolov8tracker import yolov8Tracker

VIDEO_PATH = 'D:/yolov8-2.0-train-image-video/ultralytics/LSTM+Tracking/data/cam_test/s4/10.mp4'
TRACKING_OUTPUT_PATH = 'D:/yolov8-2.0-train-image-video/ultralytics/LSTM+Tracking/data/data_txt/cam_test/s4'  # 轨迹数据保存路径

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))

# 保存轨迹为 txt 文件，文件名中增加视频名称以避免覆盖
def save_trajectory_to_txt(video_name, track_id, trajectory_points, output_dir=TRACKING_OUTPUT_PATH):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f'trajectory_{video_name}_{track_id}.txt')
    np.savetxt(file_path, trajectory_points, fmt='%.6f', delimiter=' ', header="x y", comments="")

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    # 提取视频文件名称（不带扩展名）
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    v8Tracker = yolov8Tracker()
    object_trails = {}

    while True:
        ret, im = capture.read()
        if not ret or im is None:
            break

        detections = Detections()
        output_image_frame, list_bboxs = v8Tracker.track(im)

        for item_bbox in list_bboxs:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            detections.add((x1, y1, x2, y2), None, None, track_id)

        # 更新每个跟踪对象的轨迹信息
        for xyxy, _, _, track_id in detections.detections:
            x1, y1, x2, y2 = xyxy
            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            if track_id in object_trails:
                object_trails[track_id].append((center.x, center.y))
            else:
                object_trails[track_id] = [(center.x, center.y)]

        # 保存当前帧所有跟踪对象的轨迹
        for track_id, trail_points in object_trails.items():
            save_trajectory_to_txt(video_name, track_id, trail_points)

        # 删除不在当前帧中的跟踪对象，防止轨迹无限增长
        current_ids = [item[3] for item in detections.detections]
        for tracker_id in list(object_trails.keys()):
            if tracker_id not in current_ids:
                object_trails.pop(tracker_id)

    capture.release()
