import numpy as np
import cv2
import torch  
from yolov8tracker import yolov8Tracker
from model import TrajectoryLSTM
from sklearn.preprocessing import MinMaxScaler  # Add this import for scaling

VIDEO_PATH = './visualization_video/video/MOT16-02.mp4'
RESULT_PATH = './visualization_video/output/MOT16-02.mp4'
LSTM_MODEL_PATH = './models/best_sm_model.pth' 

# Set the best parameters (update these based on your training results)
WINDOW_SIZE = 8
BEST_HIDDEN_SIZE = 128  # Update based on best result
BEST_DROPOUT = 0.3  # Update based on best result

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

def draw_predictions(output_image_frame, prediction_points, color=(255, 0, 0)):
    for i in range(len(prediction_points) - 1):
        cv2.line(output_image_frame,
                 (int(prediction_points[i][0]), int(prediction_points[i][1])),
                 (int(prediction_points[i + 1][0]), int(prediction_points[i + 1][1])),
                 color, thickness=2)

if __name__ == '__main__':
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    capture = cv2.VideoCapture(VIDEO_PATH)
    videoWriter = None
    fps = int(capture.get(5))
    print('fps:', fps)

    # Dictionary to store the trail points of each object
    object_trails = {}

    v8Tracker = yolov8Tracker()

    # load LSTM model
    model = TrajectoryLSTM(input_size=2, hidden_size=BEST_HIDDEN_SIZE, dropout_rate=BEST_DROPOUT)
    try:
        state_dict = torch.load(LSTM_MODEL_PATH)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: LSTM model file not found at {LSTM_MODEL_PATH}")
        exit()
    model.eval()

    # Initialize scaler for trajectory data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    frame_counter = 0  # add frame counter
    prediction_trails = {}  # store prediciton trajectory

    while True:
        _, im = capture.read()
        if im is None:
            break

        detections = Detections()
        output_image_frame, list_bboxs = v8Tracker.track(im)

        for item_bbox in list_bboxs:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            detections.add((x1, y1, x2, y2), confidence, class_label, track_id)

        # Add the current object's position to the trail
        for xyxy, _, _, track_id in detections.detections:
            x1, y1, x2, y2 = xyxy
            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)

            if track_id in object_trails:
                object_trails[track_id].append((center.x, center.y))
            else:
                object_trails[track_id] = [(center.x, center.y)]

            # The sliding window processes the trajectory data and passes it individually to the LSTM model
            if len(object_trails[track_id]) >= WINDOW_SIZE:
                window_data = object_trails[track_id][-WINDOW_SIZE:]
                # Normalize the window data using the scaler
                scaled_window_data = scaler.fit_transform(window_data)
                input_tensor = torch.tensor(scaled_window_data, dtype=torch.float32).unsqueeze(0)  # (1, WINDOW_SIZE, 2)
                with torch.no_grad():
                    prediction = model(input_tensor).squeeze(0).numpy()  # (WINDOW_SIZE, 2)
                # Reverse normalization for predictions
                prediction = scaler.inverse_transform(prediction.reshape(-1, 2))  # Ensure 2D shape

                # Append predictions to prediction_trails
                if track_id not in prediction_trails:
                    prediction_trails[track_id] = []
                prediction_trails[track_id].extend(prediction.tolist())  # Append predictions

        
        trail_colors = [(255, 0, 255)] * len(object_trails)  
        draw_trail(output_image_frame, list(object_trails.values()), trail_colors)

        
        for track_id, prediction_trail in prediction_trails.items():
            prediction_trail = np.clip(prediction_trail, [0, 0], [width, height])
            draw_predictions(output_image_frame, prediction_trail, color=(255, 0, 0)) 

        # Remove trails of objects that are not detected in the current frame
        for tracker_id in list(object_trails.keys()):
            if tracker_id not in [item[3] for item in detections.detections]:
                object_trails.pop(tracker_id)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0])
            )

        videoWriter.write(output_image_frame)
        cv2.imshow('output', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
































