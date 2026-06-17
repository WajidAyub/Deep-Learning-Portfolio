import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Load CIFAR-10 locally
import os, pickle
def load_cifar10_local(path):
    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            d_decoded = {k.decode('utf8'): v for k, v in d.items()}
        data = d_decoded['data']
        labels = d_decoded['labels']
        data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        return data, labels
    x_train = np.empty((50000, 32, 32, 3), dtype='uint8')
    y_train = np.empty((50000,), dtype='uint8')
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(path, f'data_batch_{i}'))
        x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels
    x_test, y_test = load_batch(os.path.join(path, 'test_batch'))
    return (x_train, y_train), (x_test, np.array(y_test))

local_path = os.path.expanduser('~/.keras/datasets/cifar-10-batches-py')
(x_train_full, y_train_full), (x_test_full, y_test_full) = load_cifar10_local(local_path)

# Select 3 classes: automobile(1)=Plastic, ship(8)=Paper, truck(9)=Metal
selected_classes = [1, 8, 9]
class_names = ['Plastic', 'Paper', 'Metal']

# Filter training data
train_mask = np.isin(y_train_full.flatten(), selected_classes)
x_train = x_train_full[train_mask]
y_train = y_train_full[train_mask].flatten()

# Filter test data
test_mask = np.isin(y_test_full.flatten(), selected_classes)
x_test = x_test_full[test_mask]
y_test = y_test_full[test_mask].flatten()

# Remap labels: 1→0 (Plastic), 8→1 (Paper), 9→2 (Metal)
label_map = {1: 0, 8: 1, 9: 2}
y_train = np.array([label_map[y] for y in y_train])
y_test = np.array([label_map[y] for y in y_test])

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples:     {x_test.shape[0]}")
print(f"Image shape:      {x_train.shape[1:]}")
print(f"Classes:          {class_names}")

# Visualize some samples from each class
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for row, cls in enumerate([0, 1, 2]):
    idxs = np.where(y_train == cls)[0][:5]
    for col, idx in enumerate(idxs):
        axes[row, col].imshow(x_train[idx])
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_title(class_names[cls], fontsize=14, fontweight='bold')
plt.suptitle('Sample Images from Each Waste Category', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.close()

# ===== GIVEN BASE MODEL (as provided in the assignment) =====
model_base = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_base.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_base.summary()

# Train the base model for 10 epochs
history_base = model_base.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)

# Plot Training vs Validation Accuracy
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history_base.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
plt.plot(history_base.history['val_accuracy'], 'r-o', label='Validation Accuracy', linewidth=2)
plt.title('Task 1: Training vs Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history_base.history['loss'], 'b-o', label='Training Loss', linewidth=2)
plt.plot(history_base.history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
plt.title('Task 1: Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_base_model.png')
plt.close()

# Print final numbers
train_acc_base = history_base.history['accuracy'][-1]
val_acc_base = history_base.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy:   {train_acc_base:.4f}")
print(f"Final Validation Accuracy: {val_acc_base:.4f}")
print(f"Gap (Train - Val):         {train_acc_base - val_acc_base:.4f}")

# ===== MODIFIED MODEL (with Dropout, BatchNorm, more filters, lower LR) =====
model_modified = tf.keras.Sequential([
    # First Conv Block — increased filters from 32 to 64, added BatchNorm
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Second Conv Block — increased filters from 64 to 128, added BatchNorm
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),     # <-- Added Dropout to reduce overfitting
    tf.keras.layers.Dense(3, activation='softmax')
])

# Changed optimizer with reduced learning rate
model_modified.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_modified.summary()

# Train the modified model for 10 epochs
history_modified = model_modified.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)

# Plot comparison: Original vs Modified model
plt.figure(figsize=(14, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
plt.plot(history_base.history['accuracy'], 'b--', label='Original Train', linewidth=2)
plt.plot(history_base.history['val_accuracy'], 'r--', label='Original Val', linewidth=2)
plt.plot(history_modified.history['accuracy'], 'b-o', label='Modified Train', linewidth=2)
plt.plot(history_modified.history['val_accuracy'], 'r-o', label='Modified Val', linewidth=2)
plt.title('Task 2: Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss comparison
plt.subplot(1, 2, 2)
plt.plot(history_base.history['loss'], 'b--', label='Original Train Loss', linewidth=2)
plt.plot(history_base.history['val_loss'], 'r--', label='Original Val Loss', linewidth=2)
plt.plot(history_modified.history['loss'], 'b-o', label='Modified Train Loss', linewidth=2)
plt.plot(history_modified.history['val_loss'], 'r-o', label='Modified Val Loss', linewidth=2)
plt.title('Task 2: Loss Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task2_modified_model.png')
plt.close()

# Record results in a table
train_acc_mod = history_modified.history['accuracy'][-1]
val_acc_mod = history_modified.history['val_accuracy'][-1]

print("\n" + "="*50)
print(f"{'Model Version':<20} {'Train Acc':>10} {'Val Acc':>10}")
print("="*50)
print(f"{'Original':<20} {train_acc_base:>10.4f} {val_acc_base:>10.4f}")
print(f"{'Modified':<20} {train_acc_mod:>10.4f} {val_acc_mod:>10.4f}")
print("="*50)

# Resize images from 32x32 to 96x96 for MobileNetV2
x_train_resized = tf.image.resize(x_train, (96, 96)).numpy()
x_test_resized = tf.image.resize(x_test, (96, 96)).numpy()

print(f"Resized training images shape: {x_train_resized.shape}")
print(f"Resized test images shape:     {x_test_resized.shape}")

# ===== TRANSFER LEARNING MODEL: MobileNetV2 =====

# Load MobileNetV2 pretrained on ImageNet (without the top classification layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze ALL base layers — we don't want to retrain them
base_model.trainable = False

# Build the full model with our custom classification head
model_transfer = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_transfer.summary()

# Train the transfer learning model for 5 epochs
history_transfer = model_transfer.fit(
    x_train_resized, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test_resized, y_test),
    verbose=1
)

# Plot Transfer Learning results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_transfer.history['accuracy'], 'g-o', label='Train Accuracy', linewidth=2)
plt.plot(history_transfer.history['val_accuracy'], 'm-o', label='Val Accuracy', linewidth=2)
plt.title('Task 3: MobileNetV2 Transfer Learning — Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_transfer.history['loss'], 'g-o', label='Train Loss', linewidth=2)
plt.plot(history_transfer.history['val_loss'], 'm-o', label='Val Loss', linewidth=2)
plt.title('Task 3: MobileNetV2 Transfer Learning — Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_mobilenet_frozen.png')
plt.close()

# Record validation accuracy
val_acc_transfer = history_transfer.history['val_accuracy'][-1]
print(f"\nMobileNetV2 Transfer Learning — Validation Accuracy: {val_acc_transfer:.4f}")

# Unfreeze the last 10 layers of the base model for fine-tuning
base_model.trainable = True

# Freeze all layers EXCEPT the last 10
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Count trainable vs non-trainable
trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
frozen_count = sum(1 for layer in base_model.layers if not layer.trainable)
print(f"Base model layers: {len(base_model.layers)}")
print(f"Trainable layers:  {trainable_count}")
print(f"Frozen layers:     {frozen_count}")

# Re-compile with a MUCH LOWER learning rate for fine-tuning
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very small LR!
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train for 3 more epochs (fine-tuning)
history_finetune = model_transfer.fit(
    x_train_resized, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test_resized, y_test),
    verbose=1
)

# Plot Fine-Tuning results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_finetune.history['accuracy'], 'c-o', label='Train Accuracy', linewidth=2)
plt.plot(history_finetune.history['val_accuracy'], 'y-o', label='Val Accuracy', linewidth=2)
plt.axhline(y=val_acc_transfer, color='gray', linestyle='--', label=f'Pre-Finetune Val ({val_acc_transfer:.4f})')
plt.title('Task 4: Fine-Tuning — Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_finetune.history['loss'], 'c-o', label='Train Loss', linewidth=2)
plt.plot(history_finetune.history['val_loss'], 'y-o', label='Val Loss', linewidth=2)
plt.title('Task 4: Fine-Tuning — Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task4_mobilenet_finetune.png')
plt.close()

# Record results
val_acc_finetune = history_finetune.history['val_accuracy'][-1]
print(f"\nBefore Fine-Tuning — Validation Accuracy: {val_acc_transfer:.4f}")
print(f"After Fine-Tuning  — Validation Accuracy: {val_acc_finetune:.4f}")
improvement = val_acc_finetune - val_acc_transfer
print(f"Improvement: {improvement:+.4f}")

# Final comparison table of all models
print("\n" + "="*65)
print(f"{'MODEL COMPARISON SUMMARY':^65}")
print("="*65)
print(f"{'Model':<30} {'Train Acc':>12} {'Val Acc':>12}")
print("-"*65)
print(f"{'1. Original CNN':<30} {train_acc_base:>12.4f} {val_acc_base:>12.4f}")
print(f"{'2. Modified CNN':<30} {train_acc_mod:>12.4f} {val_acc_mod:>12.4f}")
print(f"{'3. MobileNetV2 (Frozen)':<30} {history_transfer.history['accuracy'][-1]:>12.4f} {val_acc_transfer:>12.4f}")
print(f"{'4. MobileNetV2 (Fine-Tuned)':<30} {history_finetune.history['accuracy'][-1]:>12.4f} {val_acc_finetune:>12.4f}")
print("="*65)
