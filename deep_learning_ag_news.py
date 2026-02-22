import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

dataset = load_dataset("ag_news")

print("\nDataset Structure:")
print(dataset)
print(f"\nTraining samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

print("Sample Records from Training Set:")
for idx in range(3):
    sample = dataset['train'][idx]
    print(f"\nSample {idx + 1}:")
    print(f"Text: {sample['text'][:200]}...")
    print(f"Label: {sample['label']} ({['World', 'Sports', 'Business', 'Sci/Tech'][sample['label']]})")

train_labels = [sample['label'] for sample in dataset['train']]
test_labels = [sample['label'] for sample in dataset['test']]
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

print("Class Distribution:")
train_counts = pd.Series(train_labels).value_counts().sort_index()
test_counts = pd.Series(test_labels).value_counts().sort_index()

for label_idx, label_name in enumerate(class_names):
    train_count = train_counts.get(label_idx, 0)
    test_count = test_counts.get(label_idx, 0)
    print(f"{label_name:15s} - Train: {train_count:6d} ({train_count/len(train_labels)*100:5.2f}%), Test: {test_count:5d} ({test_count/len(test_labels)*100:5.2f}%)")


MAX_VOCAB_SIZE = 10000  
MAX_SEQ_LENGTH = 128    
BATCH_SIZE = 32         
VALIDATION_SPLIT = 0.2  

print("Preprocessing Configuration:")
print(f"  Max Vocabulary Size: {MAX_VOCAB_SIZE}")
print(f"  Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Validation Split: {VALIDATION_SPLIT*100}%")

print("\nInitializing tokenizer...")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")

train_texts = [sample['text'] for sample in dataset['train']]
print(f"{len(train_texts)} training samples")
tokenizer.fit_on_texts(train_texts)

actual_vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
print(f"vocabulary size: {actual_vocab_size}")

def preprocess_text(texts):
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, 
                          padding='post', truncating='post')
    return padded


X_train_all = preprocess_text(train_texts)
y_train_all = np.array(train_labels)


test_texts = [sample['text'] for sample in dataset['test']]
X_test = preprocess_text(test_texts)
y_test = np.array(test_labels)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, 
    test_size=VALIDATION_SPLIT, 
    random_state=42,
    stratify=y_train_all
)

print(f"\nData Split Summary:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"\nInput shape: {X_train.shape}")
print(f"Label shape: {y_train.shape}")

print("Class distribution verification:")
for label_idx, label_name in enumerate(class_names):
    train_pct = np.sum(y_train == label_idx) / len(y_train) * 100
    val_pct = np.sum(y_val == label_idx) / len(y_val) * 100
    test_pct = np.sum(y_test == label_idx) / len(y_test) * 100
    print(f"{label_name:15s} - Train: {train_pct:5.2f}%, Val: {val_pct:5.2f}%, Test: {test_pct:5.2f}%")


model = Sequential([
    
    layers.Embedding(input_dim=actual_vocab_size, 
                    output_dim=100,  # Embedding dimension
                    input_length=MAX_SEQ_LENGTH,
                    name='embedding'),
    
    
    layers.GlobalAveragePooling1D(name='pooling'),
    
    
    layers.Dense(128, activation='relu', 
                name='dense_1'),
    
    
    layers.Dropout(0.3, name='dropout_1'),
    
    
    layers.Dense(64, activation='relu', 
                name='dense_2'),
    
    
    layers.Dropout(0.3, name='dropout_2'),
    
    
    layers.Dense(4, activation='softmax', 
                name='output')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.build(input_shape=(None, MAX_SEQ_LENGTH))

print("\nModel Architecture:")
model.summary()

total_params = model.count_params()
print(f"\nTotal Trainable Parameters: {total_params:,}")

EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)



history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print(f"\nFinal Training Metrics:")
print(f"  Training Loss: {train_loss[-1]:.4f}")
print(f"  Training Accuracy: {train_acc[-1]:.4f}")
print(f"  Validation Loss: {val_loss[-1]:.4f}")
print(f"  Validation Accuracy: {val_acc[-1]:.4f}")
print(f"  Epochs Run: {len(train_loss)}")


model.save('ag_news_classifier.h5')
print("\nModel saved as 'ag_news_classifier.h5'")


y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)


test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Test Accuracy: {test_accuracy:.4f}")

# Calculate per-class metrics
print("\n" + "="*80)
print("Per-Class Performance Metrics:")
print("="*80)

precision_scores = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_scores = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_scores = f1_score(y_test, y_pred, average=None, zero_division=0)

metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores
})

print(metrics_df.to_string(index=False))

# Macro and weighted averages
print("\n" + "="*80)
print("Aggregated Metrics:")
print("="*80)
macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nMacro Averages:")
print(f"  Precision: {macro_precision:.4f}")
print(f"  Recall: {macro_recall:.4f}")
print(f"  F1-Score: {macro_f1:.4f}")

print(f"\nWeighted Averages:")
print(f"  Precision: {weighted_precision:.4f}")
print(f"  Recall: {weighted_recall:.4f}")
print(f"  F1-Score: {weighted_f1:.4f}")


print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)






# Visualization 1: Training and Validation Loss & Accuracy Curves
print("\nGenerating visualization 1: Training curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
axes[0].plot(train_loss, label='Training Loss', marker='o', linewidth=2)
axes[0].plot(val_loss, label='Validation Loss', marker='s', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
axes[0].set_title('Model Loss During Training', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot accuracy
axes[1].plot(train_acc, label='Training Accuracy', marker='o', linewidth=2)
axes[1].plot(val_acc, label='Validation Accuracy', marker='s', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[1].set_title('Model Accuracy During Training', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.close()



fig, ax = plt.subplots(figsize=(10, 8))


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax, square=True)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - AG News Classification', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('News Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics by Class', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_by_class.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}, ax=ax, square=True, vmin=0, vmax=1)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Normalized Confusion Matrix (Row Percentages)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Overall Metrics
overall_metrics = [test_accuracy, macro_precision, macro_recall, macro_f1]
metric_names = ['Accuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1-Score']
colors_overall = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = axes[0, 0].bar(metric_names, overall_metrics, color=colors_overall, alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Overall Classification Metrics', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim([0, 1.1])
axes[0, 0].grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

colors_f1 = sns.color_palette("husl", len(class_names))
bars = axes[0, 1].bar(class_names, f1_scores, color=colors_f1, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
axes[0, 1].set_title('F1-Score by Category', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim([0, 1.1])
axes[0, 1].grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

test_counts_per_class = [np.sum(y_test == i) for i in range(4)]
bars = axes[1, 0].bar(class_names, test_counts_per_class, color=colors_f1, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Test Set Distribution by Class', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1, 1].axis('off')
summary_text = f"""
MODEL TRAINING SUMMARY

Dataset: AG News (Text Classification)
Classes: 4 (World, Sports, Business, Sci/Tech)

Training Configuration:
  • Max Epochs: {EPOCHS}
  • Batch Size: {BATCH_SIZE}
  • Early Stopping Patience: {EARLY_STOPPING_PATIENCE}
  • Validation Split: {VALIDATION_SPLIT*100}%

Data Split:
  • Training: {len(X_train):,} samples
  • Validation: {len(X_val):,} samples
  • Test: {len(X_test):,} samples

Model Architecture:
  • Vocabulary Size: {actual_vocab_size:,}
  • Embedding Dim: 100
  • Sequence Length: {MAX_SEQ_LENGTH}
  • Total Parameters: {total_params:,}

Performance:
  • Test Accuracy: {test_accuracy:.4f}
  • Macro Precision: {macro_precision:.4f}
  • Macro Recall: {macro_recall:.4f}
  • Macro F1-Score: {macro_f1:.4f}
"""

axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()


print(f"""
### Model Performance Analysis

1. Overall Performance:
   - The model achieved a test accuracy of {test_accuracy:.2%}, demonstrating strong
     multi-class classification capability on the AG News dataset.
   - Macro F1-score of {macro_f1:.4f} indicates balanced performance across all four news categories.

2. Per-Class Performance:
   - World News: Precision={precision_scores[0]:.4f}, Recall={recall_scores[0]:.4f}, F1={f1_scores[0]:.4f}
   - Sports: Precision={precision_scores[1]:.4f}, Recall={recall_scores[1]:.4f}, F1={f1_scores[1]:.4f}
   - Business: Precision={precision_scores[2]:.4f}, Recall={recall_scores[2]:.4f}, F1={f1_scores[2]:.4f}
   - Sci/Tech: Precision={precision_scores[3]:.4f}, Recall={recall_scores[3]:.4f}, F1={f1_scores[3]:.4f}

3. Training Dynamics:
   - Model converged in {len(train_loss)} epochs
   - Validation loss: {val_loss[-1]:.4f}
   - Training loss: {train_loss[-1]:.4f}
   - Gap between training and validation: {abs(train_loss[-1] - val_loss[-1]):.4f}
""")
