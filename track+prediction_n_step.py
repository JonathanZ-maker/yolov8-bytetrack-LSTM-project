import os
import cv2
import torch  
import numpy as np
from yolov8tracker import yolov8Tracker
from model import TrajectoryLSTM
from sklearn.preprocessing import MinMaxScaler

VIDEO_PATH = './visualization_video/video/s1_1.mp4'
RESULT_PATH = './visualization_video/output/s1_1_5.mp4'
LSTM_MODEL_PATH = './models/best_sm_model.pth' 

# Configuration parameters
WINDOW_SIZE = 8
BEST_HIDDEN_SIZE = 128  
BEST_DROPOUT = 0.3  
N_STEPS = 10  # Adjustable number of prediction steps

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []
    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))

def draw_trail(frame, trails, colors, trail_length=50):
    for pts, color in zip(trails, colors):
        if len(pts) >= 2:
            for j in range(1, len(pts)):
                cv2.line(frame,
                         (int(pts[j-1][0]), int(pts[j-1][1])),
                         (int(pts[j][0]), int(pts[j][1])),
                         color, thickness=3)
        if len(pts) > trail_length:
            pts[:] = pts[-trail_length:]

def draw_predictions(frame, pred_points, color=(255, 0, 0)):
    if len(pred_points) >= 2:
        for i in range(1, len(pred_points)):
            cv2.line(frame,
                     (int(pred_points[i-1][0]), int(pred_points[i-1][1])),
                     (int(pred_points[i][0]), int(pred_points[i][1])),
                     color, thickness=2)

if __name__ == '__main__':
    # Initialize video capture
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    print('fps:', fps)
    videoWriter = None

    object_trails = {}  # Stores actual trajectories per track

    v8Tracker = yolov8Tracker()

    # Load LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryLSTM(input_size=2, hidden_size=BEST_HIDDEN_SIZE, dropout_rate=BEST_DROPOUT).to(device)
    try:
        state_dict = torch.load(LSTM_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: LSTM model file not found at {LSTM_MODEL_PATH}")
        exit()
    model.eval()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        detections = Detections()
        output_frame, list_bboxs = v8Tracker.track(frame)

        # Update actual trajectories from detections
        for item in list_bboxs:
            x1, y1, x2, y2, class_label, confidence, track_id = item
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if track_id in object_trails:
                object_trails[track_id].append(center)
            else:
                object_trails[track_id] = [center]

        # Draw actual trajectories (trails) in purple
        colors = [(255, 0, 255)] * len(object_trails)
        draw_trail(output_frame, list(object_trails.values()), colors)

        # For each track with enough history, compute multi-step prediction
        for track_id, trail in object_trails.items():
            if len(trail) >= WINDOW_SIZE:
                window_data = np.array(trail[-WINDOW_SIZE:])
                # Create a new scaler instance for this window
                temp_scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_window = temp_scaler.fit_transform(window_data)
                current_seq = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, WINDOW_SIZE, 2)
                pred_points = []
                with torch.no_grad():
                    for _ in range(N_STEPS):
                        next_point = model(current_seq)  # (1, 2)
                        next_point_np = next_point.squeeze(0).cpu().numpy()
                        pred_points.append(next_point_np)
                        # Update current_seq: remove oldest, append predicted point
                        pred_tensor = torch.tensor(next_point_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        current_seq = torch.cat([current_seq[:, 1:, :], pred_tensor], dim=1)
                pred_points = np.array(pred_points)
                # Inverse transform using the temporary scaler
                pred_points = temp_scaler.inverse_transform(pred_points.reshape(-1, 2)).reshape(pred_points.shape)
                draw_predictions(output_frame, pred_points, color=(255, 0, 0))

        # Clean up trails for tracks not detected in current frame
        current_ids = [item[3] for item in detections.detections]
        for tid in list(object_trails.keys()):
            if tid not in current_ids:
                object_trails.pop(tid, None)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (output_frame.shape[1], output_frame.shape[0]))

        videoWriter.write(output_frame)
        cv2.imshow('Demo', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
