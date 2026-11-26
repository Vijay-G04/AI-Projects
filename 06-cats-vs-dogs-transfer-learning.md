# Cats vs Dogs Classifier (Transfer Learning)

**Level:** Beginner | **Estimated time:** 3 to 4 days | **Domain:** Deep learning (computer vision)

## What this project shows
Transfer learning with a pretrained model, image data pipelines, and fine-tuning.

## Dataset
Kaggle **Dogs vs. Cats** dataset (25,000 labeled images), or a smaller subset to train faster. Arrange images in folders by class.

## Tech stack
Python, TensorFlow/Keras (MobileNetV2), matplotlib, Gradio.

## Suggested folder structure
```
cats-vs-dogs/
├── data/
│   └── train/
│       ├── cats/
│       └── dogs/
├── src/
│   ├── train.py
│   └── predict.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Download the dataset and arrange it as `data/train/cats` and `data/train/dogs`. Remove any corrupt images.
2. Create training and validation datasets with `image_dataset_from_directory`.
3. Add light data augmentation (flip, rotation, zoom) to the pipeline.
4. Load MobileNetV2 pretrained on ImageNet without its top layer, freeze it, and add a small classification head.
5. Train for a few epochs and record accuracy curves.
6. Fine-tune: unfreeze the last blocks of the base model and train with a very low learning rate.
7. Save the model and build a Gradio app where users upload a photo.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import tensorflow as tf

IMG_SIZE = (160, 160)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=32)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
    augment,
    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=5)
model.save("models/cats_vs_dogs.keras")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Validation accuracy before and after fine-tuning.
- Training and validation accuracy curves.
- Example correct and incorrect predictions.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Compare MobileNetV2 against ResNet50 or EfficientNet.
- Add Grad-CAM to show what the model looks at.
- Extend to multi-class (for example, dog breeds).
- Convert to TensorFlow Lite.

## Resume line
Fine-tuned MobileNetV2 via transfer learning for cat/dog image classification (X% validation accuracy) and deployed it as a Gradio web app.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
