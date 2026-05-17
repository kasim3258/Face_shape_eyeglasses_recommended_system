from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

train_path = "dataset/training_set"
test_path = "dataset/testing_set"

# 🔥 FIX 1: Controlled augmentation (less distortion)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.05,
    height_shift_range=0.05
)

test_gen = ImageDataGenerator(rescale=1./255)

# 🔥 FIX 2: Higher resolution
train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

print("Class order:", train_data.class_indices)

# 🔥 PRETRAINED MODEL
base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)

# ===============================
# ✅ STEP 1: Freeze all layers
# ===============================
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔥 Training top layers...")
model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# ===============================
# ✅ STEP 2: Fine-tuning (LESS layers)
# ===============================
print("\n🔥 Fine-tuning model...")

base_model.trainable = True

# 🔥 FIX 3: Freeze more layers (better for small dataset)
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00003),  # 🔥 lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,   # 🔥 slightly more epochs
    callbacks=[early_stop, reduce_lr]
)

# 🔥 SAVE MODEL
model.save("faceshape_model.h5")

print("\n✅ Training completed!")