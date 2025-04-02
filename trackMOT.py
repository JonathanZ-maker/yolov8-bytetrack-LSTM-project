import cv2
from yolov8tracker import yolov8Tracker

# 输入和输出路径
VIDEO_PATH = 'D:/yolov8-2.0-train-image-video/ultralytics/results/MOT_results/video/MOT16-11.mp4'
OUTPUT_TXT_PATH = 'D:/yolov8-2.0-train-image-video/ultralytics/results/MOT_results/botsort/MOT16-11.txt'

def calculate_visibility_simple(x, y, w, h, frame_width, frame_height):
    """
    Calculate visibility based on whether the bounding box is partially outside the image frame.
    :param x, y, w, h: Bounding box coordinates (x, y, width, height).
    :param frame_width, frame_height: Dimensions of the image frame.
    :return: Visibility (0.0, 0.5, 1.0).
    """
    if x < 0 or y < 0 or (x + w) > frame_width or (y + h) > frame_height:
        # Partially visible (some part is out of the frame)
        return 0.5
    if x + w <= 0 or y + h <= 0 or x >= frame_width or y >= frame_height:
        # Completely out of frame
        return 0.0
    return 1.0  # Fully visible

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

    v8Tracker = yolov8Tracker()
    tracking_results = {}
    frame_number = 0

    while True:
        ret, im = capture.read()
        if not ret:
            break

        frame_number += 1
        _, list_bboxs = v8Tracker.track(im)

        for item_bbox in list_bboxs:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            # Calculate visibility using the simple method
            visibility = calculate_visibility_simple(x, y, w, h, frame_width, frame_height)

            # Save results per ID
            if track_id not in tracking_results:
                tracking_results[track_id] = []
            tracking_results[track_id].append([
                frame_number,           # 帧编号
                track_id,               # 对象 ID
                x, y, w, h,             # 检测框
                confidence,             # 置信度
                1,                      # 固定类别 ID （7 表示行人）
                visibility              # 可见性
            ])

    # 将每个 ID 的所有帧合并在一起
    sorted_results = []
    for track_id in sorted(tracking_results.keys()):  # 按 ID 排序
        sorted_results.extend(tracking_results[track_id])  # 将每个 ID 的所有帧添加到结果中

    # 保存到文件
    with open(OUTPUT_TXT_PATH, "w") as f:
        for line in sorted_results:
            f.write(",".join(map(str, line)) + "\n")

    print(f"Tracking results saved to {OUTPUT_TXT_PATH}")
    capture.release()
    cv2.destroyAllWindows()
