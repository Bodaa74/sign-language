"""
ASL Sign Language Translator - Model Training
==============================================
Architecture: MobileNetV2 (transfer learning) + custom head
Target accuracy: ≥ 95% on validation set

Why MobileNetV2?
  • Pre-trained on ImageNet → strong visual features out of the box
  • Depthwise-separable convolutions → fast inference on CPU/edge
  • Small footprint (~14 MB) → easy to deploy in FastAPI
  • Achieves >97% on ASL A–Z with fine-tuning

Usage:
  python scripts/train.py --data_dir data/processed --epochs 30
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
INITIAL_EPOCHS = 10   # Phase 1: train only the head
FINETUNE_EPOCHS = 20  # Phase 2: unfreeze top layers
LEARNING_RATE = 1e-3
FINETUNE_LR   = 1e-5
NUM_CLASSES   = 26    # A–Z
SEED          = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────

def make_generators(data_dir: str):
    """
    Build train / val / test ImageDataGenerators.
    Training uses heavy augmentation; val/test only rescale.
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.12,
        brightness_range=[0.85, 1.15],
        horizontal_flip=False,
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_aug.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, trainable_base: bool = False) -> Model:
    """
    MobileNetV2 backbone + classification head.

    Phase 1 (trainable_base=False): only the head is trained → fast convergence.
    Phase 2 (trainable_base=True):  top 50 layers of backbone unfrozen → fine-tuning.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    # Custom head
    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.40)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="ASL_MobileNetV2")


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

def make_callbacks(checkpoint_dir: str) -> list:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "logs"),
            histogram_freq=0,
        ),
        callbacks.CSVLogger(
            os.path.join(checkpoint_dir, "training_log.csv"),
            append=True,
        ),
    ]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_phase1(model, train_gen, val_gen, checkpoint_dir, epochs=INITIAL_EPOCHS):
    """Phase 1: Frozen base, train head only."""
    print("\n" + "="*60)
    print("  PHASE 1 — Training classification head (base frozen)")
    print("="*60)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=make_callbacks(checkpoint_dir),
    )
    return history


def train_phase2(model, train_gen, val_gen, checkpoint_dir, epochs=FINETUNE_EPOCHS):
    """Phase 2: Unfreeze top 50 layers of backbone for fine-tuning."""
    print("\n" + "="*60)
    print("  PHASE 2 — Fine-tuning (top backbone layers unfrozen)")
    print("="*60)

    # Unfreeze top 50 layers of MobileNetV2
    base = model.layers[1]   # second layer is the backbone
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=make_callbacks(checkpoint_dir),
    )
    return history


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, test_gen, label_map: dict, output_dir: str):
    """Generate classification report and confusion matrix."""
    print("\n🔍 Evaluating on test set …")

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = test_gen.classes

    idx_to_label = {v: k for k, v in label_map.items()}
    class_names  = [idx_to_label[i] for i in range(len(label_map))]

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n📊 Classification Report:\n")
    print(report)

    # Save report
    with open(os.path.join(output_dir, "eval_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix — ASL A–Z", fontsize=16)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"   ✅ Confusion matrix saved → {output_dir}/confusion_matrix.png")

    # Per-class accuracy summary
    acc = np.sum(y_pred == y_true) / len(y_true)
    print(f"\n🎯 Overall Test Accuracy: {acc*100:.2f}%")
    return acc


# ─────────────────────────────────────────────
# Plot Training History
# ─────────────────────────────────────────────

def plot_history(h1, h2, output_dir: str):
    acc1  = h1.history["accuracy"]
    vacc1 = h1.history["val_accuracy"]
    acc2  = h2.history["accuracy"]
    vacc2 = h2.history["val_accuracy"]

    all_acc  = acc1  + acc2
    all_vacc = vacc1 + vacc2
    epochs   = range(1, len(all_acc) + 1)
    p1_end   = len(acc1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, all_acc,  "b-o", label="Train", markersize=4)
    ax1.plot(epochs, all_vacc, "r-o", label="Val",   markersize=4)
    ax1.axvline(x=p1_end, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    loss1  = h1.history["loss"]
    vloss1 = h1.history["val_loss"]
    loss2  = h2.history["loss"]
    vloss2 = h2.history["val_loss"]
    all_loss  = loss1  + loss2
    all_vloss = vloss1 + vloss2
    ax2.plot(epochs, all_loss,  "b-o", label="Train", markersize=4)
    ax2.plot(epochs, all_vloss, "r-o", label="Val",   markersize=4)
    ax2.axvline(x=p1_end, color="gray", linestyle="--", label="Fine-tune start")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("ASL Translator — Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Training history saved → {output_dir}/training_history.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("  ASL Sign Language Translator — Model Training")
    print("=" * 60)

    # Data
    train_gen, val_gen, test_gen = make_generators(args.data_dir)
    label_map = train_gen.class_indices   # {'A': 0, 'B': 1, ...}
    num_classes = len(label_map)
    print(f"\n📌 Classes: {num_classes}  |  {label_map}")

    # Save label map
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Build model
    model = build_model(num_classes=num_classes, trainable_base=False)

    # Phase 1
    h1 = train_phase1(model, train_gen, val_gen, args.checkpoint_dir, epochs=args.phase1_epochs)

    # Phase 2
    h2 = train_phase2(model, train_gen, val_gen, args.checkpoint_dir, epochs=args.phase2_epochs)

    # Save final model
    final_path = os.path.join(args.model_dir, "asl_model_final.keras")
    model.save(final_path)
    print(f"\n💾 Final model saved → {final_path}")

    # Also export as TFLite for edge inference
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.model_dir, "asl_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"📦 TFLite model saved → {tflite_path}")

    # Evaluate
    acc = evaluate(model, test_gen, label_map, args.model_dir)
    if acc < 0.95:
        print("\n⚠️  Accuracy below 95%. Suggestions:")
        print("    1. Add more data (especially for low-accuracy classes)")
        print("    2. Increase augmentation intensity")
        print("    3. Try ResNet50V2 as backbone (heavier but more accurate)")
        print("    4. Increase phase 2 epochs or lower fine-tune LR further")

    # Plot
    plot_history(h1, h2, args.model_dir)

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Model Trainer")
    parser.add_argument("--data_dir",       default="data/processed")
    parser.add_argument("--model_dir",      default="model/saved")
    parser.add_argument("--checkpoint_dir", default="model/checkpoints")
    parser.add_argument("--phase1_epochs",  type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--phase2_epochs",  type=int, default=FINETUNE_EPOCHS)
    args = parser.parse_args()
    main(args)
