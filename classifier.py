import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications import ResNet50V2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError

# 사용자 셋팅
data_dir = 'TrashBox\TrashBox_train_dataset_subfolders'  # 데이터셋 폴더 경로를 지정해줘
saved_model_dir = 'fine_tuned_saved_model'
saved_model_file = 'model/mobilenetv2_fine_tuned.h5'
image_exts = ['.jpg', '.jpeg', '.png']
# MobileNetV2 모델 수정 및 완전 연결층 추가
def create_fine_tune_model():
    input = layers.Input(shape=(224, 224, 3))
    base_model = ResNet50V2(input_tensor=input, include_top=False, weights='imagenet')
    bm_output = base_model.output

    # 모델 정규화 및 완전 연결층 추가
    x = layers.GlobalAveragePooling2D()(bm_output)
    x = layers.Dense(25, activation="softmax")(x)
    
    fine_tuned_model = Model(input, x)
    return fine_tuned_model

# 학습률 스케줄러
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# 이미지 파일 경로 및 라벨 정보를 담은 데이터 프레임 생성
def create_dataframe(data_dir, exts):
    data = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in exts:
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as im:
                        im.verify()  # 이미지 파일 유효성 검사
                except (IOError, ValueError, UnidentifiedImageError):
                    print(f"Invalid image file '{file_path}', skipped.")
                    continue
                label = os.path.split(subdir)[-1].lower()
                data.append((file_path, label))
    return pd.DataFrame(data, columns=['filepath', 'label'])
image_files = []
labels = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # 이미지 파일 경로와 라벨 정보 저장
            img_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(img_path))
            image_files.append(img_path)
            labels.append(label)

# 데이터 프레임 생성
data = create_dataframe(data_dir, image_exts)

# 학습 데이터 증가(ImageDataGenerator)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
# 훈련 및 검증 데이터 생성
train_generator = train_datagen.flow_from_dataframe(dataframe=train_data,
                                                    x_col='filepath',
                                                    y_col='label',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='training')

validation_generator = train_datagen.flow_from_dataframe(dataframe=valid_data,
                                                         x_col='filepath',
                                                         y_col='label',
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         subset='validation')

# 모델 생성
model = create_fine_tune_model()
model.summary()

# 기존 MobileNetV2 층 동결
for layer in model.layers[:-1]:
    layer.trainable = False

# 최적화 알고리즘 변경(SGD) 및 학습률 스케줄러 콜백 추가
opt = SGD(lr=0.1, decay=0.0, momentum=0.9, nesterov=False)
lrate = LearningRateScheduler(step_decay)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
callbacks_list = [lrate, early_stopping]
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 새로운 완전 연결층(Dense layer) 학습
history = model.fit(train_generator, validation_data=validation_generator, epochs=70, verbose=1, callbacks=callbacks_list)

# 동결을 풀기 전 학습률을 낮추어 미세조정을 효과적으로 수행
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 기존 MobileNetV2 층의 동결 해제
for layer in model.layers[-20:]:
    layer.trainable = True

# 학습률 스케줄러 콜백 추가
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate, early_stopping]

# 미세 조정(fine-tuning) 실행
history = model.fit(train_generator, validation_data=validation_generator, epochs=70, verbose=1, callbacks=callbacks_list)

model.save('model/mobilenetv2_fine_tuned.h5')
tf.saved_model.save(model, 'fine_tuned_saved_model')