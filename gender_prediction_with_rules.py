
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define the model
def build_model():
    input_layer = Input(shape=(224, 224, 3))
    
    # Shared CNN Layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Branch 1: Gender and Hair Length
    gender_output = Dense(1, activation='sigmoid', name='gender')  # Binary: 0 for male, 1 for female
    hair_output = Dense(1, activation='sigmoid', name='hair_length')  # Binary: 0 for short, 1 for long

    # Branch 2: Age Range
    age_output = Dense(3, activation='softmax', name='age_range')  # Classes: <20, 20-30, >30

    model = Model(inputs=input_layer, outputs=[gender_output, hair_output, age_output])
    return model

# Compile the model
model = build_model()
model.compile(
    optimizer='adam',
    loss={
        'gender': 'binary_crossentropy',
        'hair_length': 'binary_crossentropy',
        'age_range': 'categorical_crossentropy'
    },
    metrics={'gender': 'accuracy', 'hair_length': 'accuracy', 'age_range': 'accuracy'}
)

# Data Generators (replace with actual dataset paths)
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='multi_output',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='multi_output',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Inference with custom rules
def predict_with_rules(image_path, model):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    gender_pred, hair_pred, age_pred = model.predict(img_array)
    age_class = np.argmax(age_pred)

    # Custom rules
    if age_class == 1:  # Age range 20–30
        if hair_pred[0] > 0.5:  # Long hair
            final_gender = 'Female'
        else:  # Short hair
            final_gender = 'Male'
    else:  # Outside the age range
        final_gender = 'Female' if gender_pred[0] > 0.5 else 'Male'

    return final_gender, age_class

# Example prediction
if __name__ == "__main__":
    image_path = 'path_to_image.jpg'  # Replace with actual image path
    final_gender, age_class = predict_with_rules(image_path, model)
    print(f"Predicted Gender: {final_gender}, Age Class: {age_class}")
