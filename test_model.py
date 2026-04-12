from model import CNNModel
import cv2
import sys
import time
import numpy as np
from frames_to_opticalFlow import convertToOptical


# =========================
PATH_DATA_FOLDER = './data/'
PATH_TEST_LABEL = PATH_DATA_FOLDER + 'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'myvideo_1.mp4'
PATH_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'test_output.mp4'
PATH_COMBINED_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'combined_test_output.mp4'

MODEL_NAME = 'CNNModel_flow'
PRE_TRAINED_WEIGHTS = './best' + MODEL_NAME + '.h5'


# =========================
def predict_from_video(video_input_path, output_video_path, combined_output_path):

    predicted_labels = []

    cap = cv2.VideoCapture(video_input_path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    out_combined = cv2.VideoWriter(combined_output_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video")
        return []

    out.write(prev_frame)

    predicted_labels.append(0.0)

  
    flow_prev = [np.zeros_like(prev_frame) for _ in range(4)]

    font = cv2.FONT_HERSHEY_SIMPLEX

    t1 = time.time()
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ====== OPTICAL FLOW ======
        flow_next = convertToOptical(prev_frame, frame)

        flow_avg = (flow_prev[0] + flow_prev[1] + flow_prev[2] + flow_prev[3] + flow_next) / 4

        # update buffer
        flow_prev = [flow_next] + flow_prev[:3]


        flow_resized = cv2.resize(flow_avg, (320, 240))

        flow_norm = cv2.normalize(
            flow_resized,
            None,
            alpha=-1,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )

        model_input = flow_norm.reshape(1, 240, 320, 3)

     
        pred = float(model.predict(model_input, verbose=0)[0][0])

 
        pred = max(0, min(150, pred))

      
        if len(predicted_labels) > 0:
            pred = 0.7 * predicted_labels[-1] + 0.3 * pred

        predicted_labels.append(pred)

        speed_text = str(int(pred)) + " km/h"

        cv2.putText(frame, speed_text, (50, 60),
                    font, 1.5, (255, 255, 255), 3)


        combined = cv2.resize(flow_avg, (width, height)).astype('uint8')
        cv2.putText(combined, speed_text, (50, 60),
                    font, 1.5, (255, 255, 255), 3)

        out.write(frame)
        out_combined.write(combined)

        prev_frame = frame

        count += 1
        sys.stdout.write(f"\rProcessed: {count}/{num_frames}")

    cap.release()
    out.release()
    out_combined.release()

    print("\nPrediction completed!")
    print("Time:", time.time() - t1)

    # fix first value
    if len(predicted_labels) > 1:
        predicted_labels[0] = predicted_labels[1]

    return predicted_labels


# =========================
if __name__ == '__main__':

    print("Loading model...")

    model = CNNModel()
    model.load_weights(PRE_TRAINED_WEIGHTS)

    print("Running prediction...")

    preds = predict_from_video(
        PATH_TEST_VIDEO,
        PATH_TEST_VIDEO_OUTPUT,
        PATH_COMBINED_TEST_VIDEO_OUTPUT
    )

    print("Saving results...")

    with open(PATH_TEST_LABEL, "w") as f:
        for p in preds:
            f.write(str(p) + "\n")

    print("DONE")