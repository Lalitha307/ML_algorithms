import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

angry_dir = os.path.join(r'F:\face\angry')
happy_dir = os.path.join(r'F:\face\happy')
neutral_dir = os.path.join(r'F:\face\neutral')
sad_dir = os.path.join(r'F:\face\sad')
surprise_dir = os.path.join(r'F:\face\surprise')



train_happy_names = os.listdir(happy_dir)
print(train_happy_names[:5])


batch_size = 16

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'F:\face',  
        target_size=(48, 48),  
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['Angry','Happy','Neutral','Sad','Surprise'],
        class_mode='categorical')

target_size=(48,48)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 48*48 with 3 bytes color

     # The first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 64 neuron in the fully-connected layer
    tf.keras.layers.Dense(64, activation='relu'),
    
    
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5,activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])#RMSprop(lr=0.001)
# Total sample count
total_sample=train_generator.n
# Training
num_epochs = 25
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("E:\ML_programs\saved_models_ML\model_FD.h5")
print("Saved model to disk")
