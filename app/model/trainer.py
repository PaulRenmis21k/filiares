"""
Potato Disease Classifier - Model Training Script
=================================================
Este script descarga el dataset de hojas de papa desde Kaggle,
entrena un modelo CNN con transfer learning (MobileNetV2) y guarda el modelo.

Clases:
    - Potato___Early_blight (Tizón temprano)
    - Potato___Late_blight  (Tizón tardío)
    - Potato___healthy      (Saludable)

Arquitectura:
    - Base: MobileNetV2 (pre-trained on ImageNet, frozen)
    - Top: GlobalAveragePooling2D + Dropout + Dense(3, softmax)
    - Data augmentation se aplica SOLO en el pipeline de entrenamiento (tf.data)
"""

import os
import sys
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- CONFIGURATION --------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42
MODEL_PATH = Path("models/potato_disease_model.keras")
BEST_CHECKPOINT_PATH = Path("models/best_checkpoint.keras")
PLOTS_DIR = Path("models/plots")
AUGMENTATION = True

# Classes
CLASS_DIR_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
CLASSES = CLASS_DIR_NAMES
CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]
CLASS_IDS = ["early_blight", "late_blight", "healthy"]

# Subcarpeta dentro del dataset de Kaggle que contiene las clases
DATASET_SUBDIR = "PlantVillage"
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "training_data"))

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def download_dataset() -> Path:
    """Download the potato dataset from Kaggle using kagglehub."""
    print("[INFO] Downloading potato dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("arajmishra/potato-dataset")
    path = Path(dataset_path)
    print(f"[INFO] Dataset downloaded to: {path}")
    return path


def load_user_feedback_data(training_data_dir: Path) -> tuple:
    """
    Load user feedback images from training_data/ directory.

    Returns:
        (images, labels) as numpy arrays, or (None, None) if no data.
    """
    images = []
    labels = []

    for class_idx, class_id in enumerate(CLASS_IDS):
        class_dir = training_data_dir / class_id
        if not class_dir.exists():
            continue

        # Find all image files
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
            image_paths.extend(class_dir.glob(ext))

        for img_path in image_paths:
            try:
                img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0

                # Convert grayscale to RGB
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]

                images.append(img_array)
                label = np.zeros(len(CLASS_IDS), dtype=np.float32)
                label[class_idx] = 1.0
                labels.append(label)

            except Exception as e:
                print(f"[WARNING] Skipping {img_path}: {e}")

    if not images:
        return None, None

    print(f"[INFO] Loaded {len(images)} user feedback images")
    for i, cid in enumerate(CLASS_IDS):
        count = len(list((training_data_dir / cid).glob("*"))) if (training_data_dir / cid).exists() else 0
        if count > 0:
            print(f"       - {cid}: {count} images")

    return np.array(images), np.array(labels)


def apply_augmentation(image, label):
    """Apply data augmentation to training images.
    Operates on normalized [-1, 1] values.
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random rotation (small angle)
    image = tf.image.rot90(image, k=tf.random.uniform([], maxval=4, dtype=tf.int32))
    # Random brightness/contrast
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Clip to valid range
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def load_and_preprocess_dataset(dataset_path: Path):
    """Load images, split into train/val/test sets."""
    print("[INFO] Loading and preprocessing dataset...")

    data_dir = dataset_path / DATASET_SUBDIR
    print(f"[INFO] Using data directory: {data_dir}")
    print(f"[INFO] Subdirectories: {[d.name for d in data_dir.iterdir() if d.is_dir()]}")

    # MobileNetV2 expects values in [-1, 1]
    full_dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
        label_mode="categorical",
    )

    # Get class names BEFORE applying map/prefetch
    class_names = full_dataset.class_names
    print(f"[INFO] Classes found: {class_names}")

    # Normalize pixel values to [-1, 1] for MobileNetV2
    full_dataset = full_dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1.0, y)
    )
    full_dataset = full_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Calculate split sizes
    total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
    val_size = int(total_batches * VALIDATION_SPLIT)
    test_size = max(1, int(total_batches * TEST_SPLIT))
    train_size = total_batches - val_size - test_size

    print(f"[INFO] Total batches: {total_batches}")
    print(f"[INFO] Training batches: {train_size}")
    print(f"[INFO] Validation batches: {val_size}")
    print(f"[INFO] Test batches: {test_size}")

    # Split dataset
    train_dataset = full_dataset.take(train_size)
    remaining = full_dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)

    # Apply augmentation ONLY to training dataset
    if AUGMENTATION:
        # Unbatch, apply augmentation individually, then rebatch
        train_dataset = (
            train_dataset.unbatch()
            .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    return train_dataset, val_dataset, test_dataset, class_names


def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=3):
    """
    Build a model using MobileNetV2 transfer learning.

    Arquitectura LIMPIA:
        - MobileNetV2 base (pre-trained on ImageNet, frozen)
        - Global Average Pooling (via pooling='avg' in MobileNetV2)
        - Dropout (0.5)
        - Dense output layer with softmax

    NOTA: No se incluyen capas de aumentación en el modelo.
    La aumentación se aplica en el pipeline de tf.data.
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # Pre-trained base: MobileNetV2 (frozen)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    # Forward pass through base
    x = base_model(inputs, training=False)

    # Classification head
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="potato_disease_classifier"
    )
    return model, base_model


def plot_training_history(history, save_path: Path, title_suffix=""):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
    axes[0].set_title(f"Model Accuracy {title_suffix}", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", marker="o")
    axes[1].plot(history.history["val_loss"], label="Val Loss", marker="o")
    axes[1].set_title(f"Model Loss {title_suffix}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_path / f"training_history{title_suffix.replace(' ', '_')}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"[INFO] Training plot saved")


def evaluate_model(model, test_dataset, title="Test"):
    """Evaluate model on test set and print metrics."""
    print(f"\n[INFO] Evaluating model on {title} set...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    print(f"[INFO] {title} Accuracy: {test_acc:.4f}")
    print(f"[INFO] {title} Loss: {test_loss:.4f}")

    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(f"\n[INFO] Classification Report ({title}):")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    cm = confusion_matrix(y_true, y_pred)
    print(f"[INFO] Confusion Matrix ({title}):")
    print(cm)

    return test_acc, test_loss


def verify_loaded_model(model_path: Path, test_dataset, expected_acc: float):
    """Verify that the loaded model produces similar results to the in-memory model."""
    print(f"\n[INFO] Verifying loaded model from {model_path}...")
    loaded_model = keras.models.load_model(str(model_path))
    load_loss, load_acc = loaded_model.evaluate(test_dataset, verbose=1)
    print(f"[INFO] Loaded model - Accuracy: {load_acc:.4f}, Loss: {load_loss:.4f}")
    print(f"[INFO] In-memory model - Accuracy: {expected_acc:.4f}")
    diff = abs(load_acc - expected_acc)
    if diff < 0.05:
        print(f"[OK] Models are consistent (diff={diff:.4f})")
    else:
        print(f"[WARNING] Models differ significantly (diff={diff:.4f})")
    return load_acc, load_loss


def main():
    """Main training pipeline with transfer learning."""
    print("=" * 60)
    print("  POTATO DISEASE CLASSIFIER - TRANSFER LEARNING")
    print("  Base: MobileNetV2 (pre-trained on ImageNet)")
    print("=" * 60)

    # Step 1: Download dataset
    dataset_path = download_dataset()

    # Step 2: Load and preprocess
    train_dataset, val_dataset, test_dataset, class_names = load_and_preprocess_dataset(
        dataset_path
    )

    # Step 3: Load user feedback data and merge with training dataset
    user_images, user_labels = load_user_feedback_data(TRAINING_DATA_DIR)

    if user_images is not None:
        print(f"\n[INFO] Merging {len(user_images)} user feedback images into training data...")
        user_dataset = tf.data.Dataset.from_tensor_slices((user_images, user_labels))
        user_dataset = user_dataset.batch(BATCH_SIZE)

        # Apply augmentation to user data too
        if AUGMENTATION:
            user_dataset = (
                user_dataset.unbatch()
                .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )

        # Merge with original training dataset
        train_dataset = train_dataset.concatenate(user_dataset)
        # Re-shuffle merged dataset
        train_dataset = train_dataset.shuffle(
            buffer_size=1000, seed=RANDOM_SEED
        ).prefetch(tf.data.AUTOTUNE)

        print(f"[INFO] Combined training dataset ready")

    # Step 4: Build model with MobileNetV2
    print("\n[INFO] Building model with MobileNetV2 transfer learning...")
    model, base_model = build_model()
    model.summary()

    # Step 5: Train the model (single phase, base frozen)
    print("\n" + "=" * 60)
    print("  TRAINING PHASE: Training top layers (base frozen)")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Model checkpoint to save the best model during training
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(BEST_CHECKPOINT_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            checkpoint_cb,
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=4, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=1,
    )
    plot_training_history(history, PLOTS_DIR, title_suffix="Training")

    # Step 6: Evaluate in-memory model (with restore_best_weights)
    test_acc, test_loss = evaluate_model(model, test_dataset, title="Test")

    # Step 7: Save the final model
    model.save(MODEL_PATH)
    print(f"[INFO] Final model saved to {MODEL_PATH}")

    # Step 8: Verify loaded model consistency
    loaded_acc, loaded_loss = verify_loaded_model(MODEL_PATH, test_dataset, test_acc)

    # Step 9: Also verify the best checkpoint
    checkpoint_acc, checkpoint_loss = verify_loaded_model(
        BEST_CHECKPOINT_PATH, test_dataset, test_acc
    )

    # Use the model with better accuracy
    if checkpoint_acc > loaded_acc:
        print(f"\n[INFO] Checkpoint model is better ({checkpoint_acc:.4f} vs {loaded_acc:.4f})")
        print(f"[INFO] Copying checkpoint to final model path...")
        import shutil
        shutil.copy2(str(BEST_CHECKPOINT_PATH), str(MODEL_PATH))
        print(f"[INFO] Final model updated from checkpoint")
        final_acc = checkpoint_acc
    else:
        print(f"\n[INFO] Using final saved model ({loaded_acc:.4f})")
        final_acc = loaded_acc

    # Step 10: Save summary
    summary_path = MODEL_PATH.parent / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("POTATO DISEASE CLASSIFIER - TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write("Architecture: MobileNetV2 (transfer learning)\n")
        f.write(f"Input Size: {IMG_HEIGHT}x{IMG_WIDTH}x3 ([-1, 1] normalized)\n")
        f.write(f"Number of Classes: {len(CLASSES)}\n")
        f.write(f"Classes: {', '.join(CLASS_LABELS)}\n\n")
        f.write(f"Epochs trained: {len(history.history['loss'])}\n")
        f.write(f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Test Accuracy: {final_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Hyperparameters:\n")
        f.write("  - Base Model: MobileNetV2 (ImageNet weights, frozen)\n")
        f.write("  - Optimizer: Adam (lr=0.001)\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Data Augmentation: {AUGMENTATION} (via tf.data pipeline)\n")
        f.write(f"  - Validation Split: {VALIDATION_SPLIT}\n")
        f.write(f"  - Test Split: {TEST_SPLIT}\n")

    print(f"\n[INFO] Training summary saved to {summary_path}")
    print(f"[INFO] Training completed successfully!")
    print(f"[INFO] Final test accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
