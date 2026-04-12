from model import CNNModel

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
PATH_DATA_FOLDER = './data/'
PATH_TRAIN_LABEL = os.path.join(PATH_DATA_FOLDER, 'train.txt')
PATH_TRAIN_IMAGES_FOLDER = os.path.join(PATH_DATA_FOLDER, 'train_images')
PATH_TRAIN_IMAGES_FLOW_FOLDER = os.path.join(PATH_DATA_FOLDER, 'train_images_flow')

MODEL_NAME = 'CNNModel_flow'
WEIGHTS_PATH = f'best{MODEL_NAME}.h5'


# =========================
BATCH_SIZE = 32        # 🔥 mazāks = labāk mācās
EPOCH = 20            # 🔥 vairāk trenē
LEARNING_RATE = 1e-5  # 🔥 ļoti svarīgi (smalka apmācība)


# =========================
def prepareData(labels_path, images_path, flow_images_path):
    train_labels = []
    train_images_pair_paths = []

    with open(labels_path) as f:
        labels = f.read().split()

    for i in range(len(labels)):

        if i < 3:
            continue

        speed = float(labels[i])

        img_path = os.path.join(images_path, f"{i}.jpg")
        f1 = os.path.join(flow_images_path, f"{i-3}.jpg")
        f2 = os.path.join(flow_images_path, f"{i-2}.jpg")
        f3 = os.path.join(flow_images_path, f"{i-1}.jpg")
        f4 = os.path.join(flow_images_path, f"{i}.jpg")

        if not (os.path.exists(img_path) and os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3) and os.path.exists(f4)):
            continue

        train_images_pair_paths.append((img_path, f1, f2, f3, f4))
        train_labels.append(speed)

    return train_images_pair_paths, train_labels


# =========================
def generatorData(samples, batch_size=32):
    num_samples = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for paths, measurement in batch_samples:
                _, f1, f2, f3, f4 = paths

                flow = (
                    cv2.imread(f1) +
                    cv2.imread(f2) +
                    cv2.imread(f3) +
                    cv2.imread(f4)
                ) / 4

                flow = cv2.resize(flow, (320, 240))

                flow = cv2.normalize(
                    flow,
                    None,
                    alpha=-1,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F
                )

                images.append(flow)
                angles.append(measurement)

                # 🔁 augmentācija
                images.append(cv2.flip(flow, 1))
                angles.append(measurement)

            yield np.array(images), np.array(angles)


# =========================
if __name__ == '__main__':

    print("Loading data...")

    image_paths, labels = prepareData(
        PATH_TRAIN_LABEL,
        PATH_TRAIN_IMAGES_FOLDER,
        PATH_TRAIN_IMAGES_FLOW_FOLDER
    )

    samples = list(zip(image_paths, labels))

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print(f"Total samples: {len(samples)}")
    print(f"Train: {len(train_samples)}")
    print(f"Validation: {len(validation_samples)}")

    train_gen = generatorData(train_samples, batch_size=BATCH_SIZE)
    val_gen = generatorData(validation_samples, batch_size=BATCH_SIZE)

    print("Building model...")

    model = CNNModel()


    if os.path.exists(WEIGHTS_PATH):
        print("Loading previous weights...")
        model.load_weights(WEIGHTS_PATH)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(
            WEIGHTS_PATH,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    print("Training...")

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_samples)//BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(validation_samples)//BATCH_SIZE,
        epochs=EPOCH,
        callbacks=callbacks,
        verbose=1
    )

    print("Training complete!")


    # =========================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig('graph.png')
    plt.show()