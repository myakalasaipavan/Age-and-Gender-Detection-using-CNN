# Age-and-Gender-Detection-using-CNN
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
%matplotlib inline

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import os
from tqdm.notebook import tqdm
from PIL import Image
import numpy as np

# Define the directory path
BASE_DIR = r"C:\Age and Gender Detection\UTKFace"


# Initialize lists to store data
image_paths = []
age_labels = []
gender_labels = []
images = []

# Iterate over files in the dataset directory
for filename in tqdm(os.listdir(BASE_DIR)):
    try:
        # Full path of the image
        image_path = os.path.join(BASE_DIR, filename)
        
        # Split the filename to extract age and gender
        temp = filename.split('_')
        
        # Check that filename contains at least 'age' and 'gender'
        if len(temp) >= 2 and temp[0].isdigit() and temp[1].isdigit():
            age = int(temp[0])
            gender = int(temp[1])
            
            # Load the image using PIL and convert to array (optional, for processing)
            image = Image.open(image_path).convert("RGB")  # Convert to RGB if needed
            image_array = np.array(image)  # Convert image to numpy array
            
            # Append data to lists
            image_paths.append(image_path)
            age_labels.append(age)
            gender_labels.append(gender)
            images.append(image_array)  # Store image data as numpy array
            
        else:
            print(f"Skipping file '{filename}' due to unexpected format.")
    
    except (IndexError, ValueError, OSError) as e:
        print(f"Error processing file '{filename}': {e}")

# Verify loaded data
print("Total images loaded:", len(images))
print("Sample age labels:", age_labels[:5])
print("Sample gender labels:", gender_labels[:5])
  0%|          | 0/23708 [00:00<?, ?it/s]
Total images loaded: 23708
Sample age labels: [100, 100, 100, 100, 100]
Sample gender labels: [0, 0, 1, 1, 1]
# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

# map labels for gender
gender_dict = {0:'Male', 1:'Female'}
from PIL import Image
img = Image.open(df['image'][1])
plt.axis('off')
plt.imshow(img);
No description has been provided for this image
sns.distplot(df['age'])
<Axes: xlabel='age', ylabel='Density'>
No description has been provided for this image
# to display grid of images
plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

for index, file, age, gender in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[gender]}")
    plt.axis('off')
No description has been provided for this image
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image,color_mode='grayscale')
        img = img.resize((128, 128),Image.LANCZOS)
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 128, 128, 1)
    return features
X = extract_features(df['image'])
  0%|          | 0/23708 [00:00<?, ?it/s]
X.shape
(23708, 128, 128, 1)
# normalize the images
X = X/255.0
y_gender = np.array(df['gender'])
y_age = np.array(df['age'])
input_shape = (128, 128, 1)
inputs = Input((input_shape))
# convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

flatten = Flatten() (maxp_4)

# fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.4) (dense_1)
dropout_2 = Dropout(0.4) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy', 'mae'])
# train model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=22, validation_split=0.2)

# plot results for gender
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()

# plot results for age
loss = history.history['age_out_mae']
val_loss = history.history['val_age_out_mae']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training MAE')
plt.plot(epochs, val_loss, 'r', label='Validation MAE')
plt.title('Loss Graph')
plt.legend()
plt.show()
No description has been provided for this image
image_index = 10005
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray');
Original Gender: Female Original Age: 29
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step
Predicted Gender: Female Predicted Age: 27
No description has been provided for this image
image_index = 67
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray');
Original Gender: Male Original Age: 10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step
Predicted Gender: Male Predicted Age: 10
No description has been provided for this image
image_index = 8999
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray');
Original Gender: Male Original Age: 28
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step
Predicted Gender: Male Predicted Age: 27
No description has been provided for this image
image_index = 505
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray');
Original Gender: Male Original Age: 14
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step
Predicted Gender: Male Predicted Age: 14
No description has been provided for this image
image_index = 20
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray');
Original Gender: Male Original Age: 10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
Predicted Gender: Male Predicted Age: 9
No description has been provided for this image
