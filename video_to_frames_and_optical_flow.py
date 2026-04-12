import cv2
import os, time, sys, shutil
import numpy as np
from frames_to_opticalFlow import convertToOptical

PATH_DATA_FOLDER = './data/'

PATH_TRAIN_LABEL_PREPROCESSED = PATH_DATA_FOLDER +  'train_preprocessed.txt'

PATH_TRAIN_LABEL = PATH_DATA_FOLDER +  'train.txt'
PATH_TRAIN_VIDEO = PATH_DATA_FOLDER + 'myvideo.mp4'
PATH_TRAIN_FLOW_VIDEO = PATH_DATA_FOLDER + 'flow_train.mp4'
PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER +  'slaidzinasana_image/'
PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'salidzinasana_flow/'

PATH_TEST_LABEL = PATH_DATA_FOLDER +  'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_FLOW_VIDEO = PATH_DATA_FOLDER + 'flow_test.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER +  'test_images/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'test_images_flow/'


def preprocess_data(video_input_path, flow_video_output_path, image_folder_path, flow_image_folder_path, type):

    os.makedirs(image_folder_path, exist_ok=True)
    os.makedirs(flow_image_folder_path, exist_ok=True)


    existing_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]

    if len(existing_files) == 0:
        start_index = 0
    else:
        indices = [int(os.path.splitext(f)[0]) for f in existing_files]
        start_index = max(indices) + 1

    print("Starting from index:", start_index)
    print("Processing video:", video_input_path)

    video_reader = cv2.VideoCapture(video_input_path)

    num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(flow_video_output_path, fourcc, fps, frame_size)

    t1 = time.time()
    ret, prev_frame = video_reader.read()

    if not ret:
        print("Error reading video")
        return

    # pirmā bilde
    cv2.imwrite(os.path.join(image_folder_path, f"{start_index}.jpg"), prev_frame)

    count = 1
    while True:
        ret, next_frame = video_reader.read()
        if not ret:
            break

        bgr_flow = convertToOptical(prev_frame, next_frame)

        current_index = start_index + count

        image_path_out = os.path.join(image_folder_path, f"{current_index}.jpg")
        flow_image_path_out = os.path.join(flow_image_folder_path, f"{current_index}.jpg")

        cv2.imwrite(image_path_out, next_frame)
        cv2.imwrite(flow_image_path_out, bgr_flow)

        video_writer.write(bgr_flow)

        prev_frame = next_frame
        count += 1

        if count % 100 == 0:
            print(f"Processed {count} frames")

    video_reader.release()
    video_writer.release()

    print('Conversion completed!')
    print('Total frames processed:', count)
    return



if __name__ == '__main__':


    '''PREPROCESS DATA DOES 3 THINGS:
        1. Convert video to optical flow and save their respective images
        2. Augment image and optical flow data by Inverting them horizontally'''   ## NOW DONE IN TRAIN_MODEL.PY ITSELF IN GENERATOR DATA

    preprocess_data(PATH_TRAIN_VIDEO, PATH_TRAIN_FLOW_VIDEO, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER, type='train')
