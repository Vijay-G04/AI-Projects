# Plant Disease Detection from Leaf Images

**Level:** Beginner | **Estimated time:** 4 to 6 days | **Domain:** Deep learning (computer vision)

## What this project shows
Multi-class image classification with a real-world application, transfer learning, and handling many classes.

## Dataset
**PlantVillage** dataset (available on Kaggle, also in TensorFlow Datasets as `plant_village`). Roughly 38 classes of healthy and diseased leaves across crops. Arrange as one folder per class.

## Tech stack
Python, TensorFlow/Keras (MobileNetV2), matplotlib, scikit-learn (for metrics), Gradio.

## Suggested folder structure
```
plant-disease/
├── data/
│   └── PlantVillage/
│       ├── Apple___healthy/
│       ├── Tomato___Late_blight/
│       └── ...
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Download and inspect the dataset. Count images per class and plot the distribution.
2. Create train and validation datasets from the class folders.
3. Add augmentation (flip, rotation, brightness) to improve robustness.
4. Train MobileNetV2 with a frozen base, then fine-tune the top layers.
5. Evaluate with accuracy, a per-class report, and a confusion matrix. Find which diseases get confused.
6. Note the limitation: PlantVillage images have clean backgrounds, so real-world field photos will perform worse. Test with a few of your own photos.
7. Build a Gradio app: upload a leaf, get the top-3 predicted classes with confidence.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import tensorflow as tf

IMG_SIZE = (160, 160)
DATA_DIR = "data/PlantVillage"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=32)

class_names = train_ds.class_names
print("Classes:", len(class_names))

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names)),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=5)
model.save("models/plant_disease.keras")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Validation accuracy and a per-class report.
- A confusion matrix highlighting the most confused diseases.
- Results on your own real-world leaf photos, with commentary on the gap.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Add Grad-CAM heatmaps showing which leaf regions drove the prediction.
- Train on field-condition images to improve realism.
- Convert to TensorFlow Lite for a mobile-friendly model.
- Add treatment information for each disease in the app output.

## Resume line
Developed a 38-class plant disease classifier using transfer learning (MobileNetV2), achieving X% validation accuracy, with a web demo for leaf image uploads.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
