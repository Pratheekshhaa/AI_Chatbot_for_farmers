import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

DATASET_PATH = "data/plant_disease/test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    subset="training",
    batch_size=BATCH_SIZE
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    subset="validation",
    batch_size=BATCH_SIZE
)

base = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base.trainable = False

x = Flatten()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training model... This will take 2–10 minutes depending on CPU.")
model.fit(train_gen, validation_data=val_gen, epochs=5)

model.save("disease_model.h5")
print("Model saved successfully as disease_model.h5!")
print("Total classes:", train_gen.class_indices)
