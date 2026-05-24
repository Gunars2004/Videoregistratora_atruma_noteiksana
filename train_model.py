# # # from model2 import CNNModel
# # from model import CNNModel

# # import cv2
# # import numpy as np
# # import os, sys
# # from os import listdir
# # from os.path import join
# # import matplotlib.pyplot as plt

# # import sklearn
# # from sklearn.model_selection import train_test_split
# # from sklearn.utils import shuffle

# # from keras.models import Sequential
# # from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Reshape, BatchNormalization, Activation
# # from keras.callbacks import EarlyStopping, ModelCheckpoint

# # from PIL import Image
# # from frames_to_opticalFlow import convertToOptical

# # PATH_DATA_FOLDER = './data/'
# # PATH_TRAIN_LABEL = PATH_DATA_FOLDER +  'train.txt'
# # PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER +  'train_images/'
# # PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'train_images_flow/'

# # TYPE_FLOW_PRECOMPUTED = 0
# # TYPE_ORIGINAL = 1

# # BATCH_SIZE = 128
# # EPOCH = 50

# # MODEL_NAME = 'CNNModel_flow'
# # # MODEL_NAME = 'CNNModel_combined'


# # def prepareData(labels_path, images_path, flow_images_path, type=TYPE_FLOW_PRECOMPUTED):
# #     num_train_labels = 0
# #     train_labels = []
# #     train_images_pair_paths = []

# #     with open(labels_path) as txt_file:
# #         labels_string = txt_file.read().split()

# #         for i in range(4, len(labels_string)):
# #             speed = float(labels_string[i])
# #             train_labels.append(speed)

# #             if type == TYPE_FLOW_PRECOMPUTED:
# #                 # Combine original and pre computed optical flow
# #                 train_images_pair_paths.append( ( os.getcwd() + images_path[1:] + str(i)+ '.jpg',  os.getcwd() + flow_images_path[1:] + str(i-3) + '.jpg',   os.getcwd() + flow_images_path[1:] + str(i-2) + '.jpg',   os.getcwd() + flow_images_path[1:] + str(i-1) + '.jpg',  os.getcwd() + flow_images_path[1:] + str(i) + '.jpg') )
# #             else:
# #                 # Combine 2 consecutive frames and calculate optical flow
# #                 train_images_pair_paths.append( ( os.getcwd() + images_path[1:] + str(i-1)+ '.jpg',  os.getcwd() + images_path[1:] + str(i) + '.jpg') )

# #     return train_images_pair_paths, train_labels


# # def generatorData(samples, batch_size=32, type=TYPE_FLOW_PRECOMPUTED):
# #     num_samples = len(samples)
# #     while 1: # Loop forever so the generator never terminates
# #         samples = sklearn.utils.shuffle(samples)
# #         for offset in range(0, num_samples, batch_size):
# #             batch_samples = samples[offset:offset+batch_size]

# #             images = []
# #             angles = []
# #             for imagePath, measurement in batch_samples:

# #                 combined_image = None
# #                 flow_image_bgr = None

# #                 if type == TYPE_FLOW_PRECOMPUTED:

# #                     # curr_image_path, flow_image_path = imagePath
# #                     # flow_image_bgr = cv2.imread(flow_image_path)
# #                     curr_image_path, flow_image_path1, flow_image_path2,flow_image_path3, flow_image_path4 = imagePath
# #                     flow_image_bgr = (cv2.imread(flow_image_path1) +cv2.imread(flow_image_path2) +cv2.imread(flow_image_path3) +cv2.imread(flow_image_path4) )/4

# #                     curr_image = cv2.imread(curr_image_path)
# #                     curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

# #                 else:
# #                     prev_image_path, curr_image_path = imagePath
# #                     prev_image = cv2.imread(prev_image_path)
# #                     curr_image = cv2.imread(curr_image_path)
# #                     flow_image_bgr = convertToOptical(prev_image, curr_image)
# #                     curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)


# #                 combined_image = 0.1*curr_image + flow_image_bgr
# #                 #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
# #                 combined_image = flow_image_bgr

# #                 combined_image = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# #                 combined_image = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)

# #                 # im = Image.fromarray(combined_image)
# #                 # plt.imshow(im)
# #                 # plt.show()

# #                 images.append(combined_image)
# #                 angles.append(measurement)

# #                 # AUGMENTING DATA
# #                 # Flipping image, correcting measurement and  measuerement

# #                 images.append(cv2.flip(combined_image,1))
# #                 angles.append(measurement)

# #             inputs = np.array(images)
# #             outputs = np.array(angles)
# #             yield sklearn.utils.shuffle(inputs, outputs)


# # if __name__ == '__main__':

# #     type_ = TYPE_FLOW_PRECOMPUTED   ## optical flow pre computed
# #     # type = TYPE_ORIGINAL

# #     train_images_pair_paths, train_labels =  prepareData(PATH_TRAIN_LABEL, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER, type=type_)

# #     samples = list(zip(train_images_pair_paths, train_labels))
# #     train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# #     print('Total Images: {}'.format( len(train_images_pair_paths)))
# #     print('Train samples: {}'.format(len(train_samples)))
# #     print('Validation samples: {}'.format(len(validation_samples)))

# #     training_generator = generatorData(train_samples, batch_size=BATCH_SIZE, type=type_)
# #     validation_generator = generatorData(validation_samples, batch_size=BATCH_SIZE, type=type_)

# #     print('Training model...')

# #     model = CNNModel()

# #     callbacks = [EarlyStopping(monitor='val_loss', patience=3),
# #              ModelCheckpoint(filepath='best'+MODEL_NAME+'.h5', monitor='val_loss', save_best_only=True)]

# #     history_object = model.fit_generator(training_generator, samples_per_epoch= \
# #                      len(train_samples)//BATCH_SIZE, validation_data=validation_generator, \
# #                      validation_steps=len(validation_samples)//BATCH_SIZE, callbacks=callbacks, epochs=EPOCH, verbose=1)

# #     print('Training model complete...')

# #     print(history_object.history.keys())
# #     print('Loss')
# #     print(history_object.history['loss'])
# #     print('Validation Loss')
# #     print(history_object.history['val_loss'])


# #     plt.figure(figsize=[10,8])
# #     plt.plot(np.arange(1, len(history_object.history['loss'])+1), history_object.history['loss'],'r',linewidth=3.0)
# #     plt.plot(np.arange(1, len(history_object.history['val_loss'])+1), history_object.history['val_loss'],'b',linewidth=3.0)
# #     plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
# #     plt.xlabel('Epochs ',fontsize=16)
# #     plt.ylabel('Loss',fontsize=16)
# #     plt.title('Loss Curves',fontsize=16)
# #     plt.show()
# #     plt.savefig('graph.png')

# # from model import CNNModel

# # import cv2
# # import numpy as np
# # import os
# # import matplotlib.pyplot as plt

# # from sklearn.model_selection import train_test_split
# # from sklearn.utils import shuffle

# # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # PATH_DATA_FOLDER = './data/'
# # PATH_TRAIN_LABEL = os.path.join(PATH_DATA_FOLDER, 'train.txt')
# # PATH_TRAIN_IMAGES_FOLDER = os.path.join(PATH_DATA_FOLDER, 'train_images')
# # PATH_TRAIN_IMAGES_FLOW_FOLDER = os.path.join(PATH_DATA_FOLDER, 'train_images_flow')

# # BATCH_SIZE = 64
# # EPOCH = 10

# # MODEL_NAME = 'CNNModel_flow'

# # # =========================

# # # =========================
# # def prepareData(labels_path, images_path, flow_images_path):
# #     train_labels = []
# #     train_images_pair_paths = []

# #     with open(labels_path) as f:
# #         labels = f.read().split()

# #     for i in range(len(labels)):
# #         speed = float(labels[i])

# #        
# #         if i < 3:
# #             continue

# #         img_path = os.path.join(images_path, f"{i}.jpg")
# #         f1 = os.path.join(flow_images_path, f"{i-3}.jpg")
# #         f2 = os.path.join(flow_images_path, f"{i-2}.jpg")
# #         f3 = os.path.join(flow_images_path, f"{i-1}.jpg")
# #         f4 = os.path.join(flow_images_path, f"{i}.jpg")

# #         if not (os.path.exists(img_path) and os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3) and os.path.exists(f4)):
# #             continue

# #         train_images_pair_paths.append((img_path, f1, f2, f3, f4))
# #         train_labels.append(speed)

# #     return train_images_pair_paths, train_labels

# # # =========================

# # # =========================
# # def generatorData(samples, batch_size=32):
# #     num_samples = len(samples)

# #     while True:
# #         shuffle(samples)

# #         for offset in range(0, num_samples, batch_size):
# #             batch_samples = samples[offset:offset+batch_size]

# #             images = []
# #             angles = []

# #             for paths, measurement in batch_samples:
# #                 curr_image_path, f1, f2, f3, f4 = paths

# #                 flow = (
# #                     cv2.imread(f1) +
# #                     cv2.imread(f2) +
# #                     cv2.imread(f3) +
# #                     cv2.imread(f4)
# #                 ) / 4

# #                 # resize uz pareizo izmēru
# #                 flow = cv2.resize(flow, (320, 240))

# #                 flow = cv2.normalize(flow, None, alpha=-1, beta=1,
# #                                      norm_type=cv2.NORM_MINMAX,
# #                                      dtype=cv2.CV_32F)

# #                 images.append(flow)
# #                 angles.append(measurement)

# #           
# #                 images.append(cv2.flip(flow, 1))
# #                 angles.append(measurement)

# #             yield np.array(images), np.array(angles)

# # # =========================

# # # =========================
# # if __name__ == '__main__':

# #     print("Loading data...")

# #     image_paths, labels = prepareData(
# #         PATH_TRAIN_LABEL,
# #         PATH_TRAIN_IMAGES_FOLDER,
# #         PATH_TRAIN_IMAGES_FLOW_FOLDER
# #     )

# #     samples = list(zip(image_paths, labels))

# #     train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# #     print(f"Total samples: {len(samples)}")
# #     print(f"Train: {len(train_samples)}")
# #     print(f"Validation: {len(validation_samples)}")

# #     train_gen = generatorData(train_samples, batch_size=BATCH_SIZE)
# #     val_gen = generatorData(validation_samples, batch_size=BATCH_SIZE)

# #     print("Building model...")

# #     model = CNNModel()

# #     callbacks = [
# #         EarlyStopping(monitor='val_loss', patience=3),
# #         ModelCheckpoint(f'best{MODEL_NAME}.h5',
# #                         monitor='val_loss',
# #                         save_best_only=True)
# #     ]

# #     print("Training...")

# #     history = model.fit(
# #         train_gen,
# #         steps_per_epoch=len(train_samples)//BATCH_SIZE,
# #         validation_data=val_gen,
# #         validation_steps=len(validation_samples)//BATCH_SIZE,
# #         epochs=EPOCH,
# #         callbacks=callbacks,
# #         verbose=1
# #     )

# #     print("Training complete!")

# #     # =========================

# #     # =========================
# #     plt.plot(history.history['loss'])
# #     plt.plot(history.history['val_loss'])
# #     plt.title('Loss')
# #     plt.ylabel('Loss')
# #     plt.xlabel('Epoch')
# #     plt.legend(['Train', 'Validation'])
# #     plt.savefig('graph.png')
# #     plt.show()
# ################################################
# from model import CNNModel

# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # =========================

# # =========================
# PATH_DATA_FOLDER = './data/'
# PATH_TRAIN_LABEL = os.path.join(PATH_DATA_FOLDER, 'train.txt')
# PATH_TRAIN_IMAGES_FOLDER = os.path.join(PATH_DATA_FOLDER, 'train_images')
# PATH_TRAIN_IMAGES_FLOW_FOLDER = os.path.join(PATH_DATA_FOLDER, 'train_images_flow')

# MODEL_NAME = 'CNNModel_flow'
# WEIGHTS_PATH = f'best{MODEL_NAME}.h5'

# # =========================

# # =========================
# BATCH_SIZE = 32       
# EPOCH = 20           
# LEARNING_RATE = 1e-5  

# # =========================

# # =========================
# def prepareData(labels_path, images_path, flow_images_path):
#     train_labels = []
#     train_images_pair_paths = []

#     with open(labels_path) as f:
#         labels = f.read().split()

#     for i in range(len(labels)):

#         if i < 3:
#             continue

#         speed = float(labels[i])

#         img_path = os.path.join(images_path, f"{i}.jpg")
#         f1 = os.path.join(flow_images_path, f"{i-3}.jpg")
#         f2 = os.path.join(flow_images_path, f"{i-2}.jpg")
#         f3 = os.path.join(flow_images_path, f"{i-1}.jpg")
#         f4 = os.path.join(flow_images_path, f"{i}.jpg")

#         if not (os.path.exists(img_path) and os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3) and os.path.exists(f4)):
#             continue

#         train_images_pair_paths.append((img_path, f1, f2, f3, f4))
#         train_labels.append(speed)

#     return train_images_pair_paths, train_labels

# # =========================

# # =========================
# def generatorData(samples, batch_size=32):
#     num_samples = len(samples)

#     while True:
#         shuffle(samples)

#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]

#             images = []
#             angles = []

#             for paths, measurement in batch_samples:
#                 _, f1, f2, f3, f4 = paths

#                 flow = (
#                     cv2.imread(f1) +
#                     cv2.imread(f2) +
#                     cv2.imread(f3) +
#                     cv2.imread(f4)
#                 ) / 4

#                 flow = cv2.resize(flow, (320, 240))

#                 flow = cv2.normalize(
#                     flow,
#                     None,
#                     alpha=-1,
#                     beta=1,
#                     norm_type=cv2.NORM_MINMAX,
#                     dtype=cv2.CV_32F
#                 )

#                 images.append(flow)
#                 angles.append(measurement)

#           
#                 images.append(cv2.flip(flow, 1))
#                 angles.append(measurement)

#             yield np.array(images), np.array(angles)

# # =========================

# # =========================
# if __name__ == '__main__':

#     print("Loading data...")

#     image_paths, labels = prepareData(
#         PATH_TRAIN_LABEL,
#         PATH_TRAIN_IMAGES_FOLDER,
#         PATH_TRAIN_IMAGES_FLOW_FOLDER
#     )

#     samples = list(zip(image_paths, labels))

#     train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#     print(f"Total samples: {len(samples)}")
#     print(f"Train: {len(train_samples)}")
#     print(f"Validation: {len(validation_samples)}")

#     train_gen = generatorData(train_samples, batch_size=BATCH_SIZE)
#     val_gen = generatorData(validation_samples, batch_size=BATCH_SIZE)

#     print("Building model...")

#     model = CNNModel()

#     
#     if os.path.exists(WEIGHTS_PATH):
#         print("Loading previous weights...")
#         model.load_weights(WEIGHTS_PATH)

#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=3),
#         ModelCheckpoint(
#             WEIGHTS_PATH,
#             monitor='val_loss',
#             save_best_only=True
#         )
#     ]

#     print("Training...")

#     history = model.fit(
#         train_gen,
#         steps_per_epoch=len(train_samples)//BATCH_SIZE,
#         validation_data=val_gen,
#         validation_steps=len(validation_samples)//BATCH_SIZE,
#         epochs=EPOCH,
#         callbacks=callbacks,
#         verbose=1
#     )

#     print("Training complete!")

#     # =========================

#     # =========================
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'])
#     plt.savefig('graph.png')
#     plt.show()
#     ##############################################################
from model import CNNModel

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================================================

# =========================================================
PATH_DATA_FOLDER = './data/'

PATH_TRAIN_LABEL = os.path.join(
    PATH_DATA_FOLDER,
    'train.txt'
)

PATH_TRAIN_IMAGES_FOLDER = os.path.join(
    PATH_DATA_FOLDER,
    'train_images'
)

PATH_TRAIN_IMAGES_FLOW_FOLDER = os.path.join(
    PATH_DATA_FOLDER,
    'train_images_flow'
)

MODEL_NAME = 'CNNModel_flow'
WEIGHTS_PATH = f'best_{MODEL_NAME}.h5'

# =========================================================

# =========================================================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-5

# =========================================================

# =========================================================
NUM_VIDEOS = 10
FRAMES_PER_VIDEO = 1800

# pēdējie 2 video validācijai
VALIDATION_VIDEOS = [8, 9]

# =========================================================

# =========================================================
def prepareData(labels_path,
                images_path,
                flow_images_path):

    labels_list = []
    image_pairs = []

    with open(labels_path) as f:
        labels = f.read().split()

    for i in range(len(labels)):

        # nepieciešami 4 optical flow frame
        if i < 3:
            continue

        speed = float(labels[i])

        current_image = os.path.join(
            images_path,
            f"{i}.jpg"
        )

        flow1 = os.path.join(
            flow_images_path,
            f"{i-3}.jpg"
        )

        flow2 = os.path.join(
            flow_images_path,
            f"{i-2}.jpg"
        )

        flow3 = os.path.join(
            flow_images_path,
            f"{i-1}.jpg"
        )

        flow4 = os.path.join(
            flow_images_path,
            f"{i}.jpg"
        )

        # pārbauda vai faili eksistē
        if not (
            os.path.exists(current_image)
            and os.path.exists(flow1)
            and os.path.exists(flow2)
            and os.path.exists(flow3)
            and os.path.exists(flow4)
        ):
            continue

        image_pairs.append(
            (
                current_image,
                flow1,
                flow2,
                flow3,
                flow4
            )
        )

        labels_list.append(speed)

    return image_pairs, labels_list

# =========================================================

# =========================================================
def generatorData(samples,
                  batch_size=32):

    num_samples = len(samples)

    while True:

        shuffle(samples)

        for offset in range(
            0,
            num_samples,
            batch_size
        ):

            batch_samples = samples[
                offset:offset + batch_size
            ]

            images = []
            speeds = []

            for paths, measurement in batch_samples:

                _, f1, f2, f3, f4 = paths

                # =================================================
                # Optical Flow Averaging
                # =================================================
                flow = (
                    cv2.imread(f1).astype(np.float32)
                    + cv2.imread(f2).astype(np.float32)
                    + cv2.imread(f3).astype(np.float32)
                    + cv2.imread(f4).astype(np.float32)
                ) / 4.0

                # =================================================
                # Resize to CNN input size
                # =================================================
                flow = cv2.resize(
                    flow,
                    (320, 240)
                )

                # =================================================
                # Normalize [-1 ; 1]
                # =================================================
                flow = cv2.normalize(
                    flow,
                    None,
                    alpha=-1,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F
                )

                images.append(flow)
                speeds.append(measurement)

                # =================================================
                # DATA AUGMENTATION
                # Horizontal Flip
                # =================================================
                flipped = cv2.flip(flow, 1)

                images.append(flipped)
                speeds.append(measurement)

            yield (
                np.array(images),
                np.array(speeds)
            )

# =========================================================

# =========================================================
if __name__ == '__main__':

    print("Loading dataset...")

    image_paths, labels = prepareData(
        PATH_TRAIN_LABEL,
        PATH_TRAIN_IMAGES_FOLDER,
        PATH_TRAIN_IMAGES_FLOW_FOLDER
    )

    # =====================================================
    # SORT FRAMES CORRECTLY
    # =====================================================
    image_paths = sorted(
        image_paths,
        key=lambda x:
        int(
            os.path.basename(
                x[0]
            ).split('.')[0]
        )
    )

    samples = list(
        zip(image_paths, labels)
    )

    # =====================================================
   
    # =====================================================
    train_samples = []
    validation_samples = []

    for sample in samples:

        image_path = sample[0][0]

        # frame numurs
        frame_number = int(
            os.path.basename(
                image_path
            ).split('.')[0]
        )

        # video ID
        video_id = frame_number // FRAMES_PER_VIDEO

        # validācijas video
        if video_id in VALIDATION_VIDEOS:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    # =====================================================
    # DATASET INFO
    # =====================================================
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")

    print(
        f"Training videos: "
        f"{NUM_VIDEOS - len(VALIDATION_VIDEOS)}"
    )

    print(
        f"Validation videos: "
        f"{len(VALIDATION_VIDEOS)}"
    )

    # =====================================================
    # GENERATORS
    # =====================================================
    train_generator = generatorData(
        train_samples,
        batch_size=BATCH_SIZE
    )

    validation_generator = generatorData(
        validation_samples,
        batch_size=BATCH_SIZE
    )

    # =====================================================
    # BUILD MODEL
    # =====================================================
    print("Building model...")

    model = CNNModel()

    # =====================================================
    # LOAD PREVIOUS WEIGHTS
    # =====================================================
    if os.path.exists(WEIGHTS_PATH):

        print("Loading previous weights...")

        model.load_weights(
            WEIGHTS_PATH
        )

    # =====================================================
    # CALLBACKS
    # =====================================================
    callbacks = [

        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=WEIGHTS_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # =====================================================
    # TRAIN MODEL
    # =====================================================
    print("Training model...")

    history = model.fit(

        train_generator,

        steps_per_epoch=
        len(train_samples) // BATCH_SIZE,

        validation_data=
        validation_generator,

        validation_steps=
        len(validation_samples) // BATCH_SIZE,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1
    )

    print("Training complete!")

    # =====================================================
  
    # =====================================================
    plt.figure(figsize=(10, 6))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Training and Validation Loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend([
        'Training Loss',
        'Validation Loss'
    ])

    plt.grid(True)

    plt.savefig('graph.png')

    plt.show()
