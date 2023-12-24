import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from matplotlib import image as mpimg

# smooth = 1e-15
# def dice_coef(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# Register the custom loss and metric functions
# with tf.keras.utils.custom_object_scope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
#     model1 = load_model('model/model_Unet.h5')
def segmentation(input_img):
    smooth = 1e-15
    def dice_coef(y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)

# Register the custom loss and metric functions
    with tf.keras.utils.custom_object_scope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        model1 = load_model('model/model_Unet.h5')

    img = image.load_img("images/" + input_img, target_size=(256, 256))  # Load the image using OpenCV
    img = np.asarray(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_new = cv2.resize(gray_img, (256, 256))
    img_scaled = img_new / 255
    img_scaled = np.expand_dims(img_scaled, axis=0)
    img_scaled = np.expand_dims(img_scaled, axis=-1)
    img_scaled = np.repeat(img_scaled, 3, axis=-1)  # Repeat along the last axis
    print()
    # Predict segmentation mask
    y_pred = model1.predict(img_scaled)
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.uint32)  # Convert to uint8

    
    # Assuming the output is a probability mask and you want to threshold it
    predicted_mask = y_pred[0]
    threshold = 0.5  # You can adjust this threshold as needed
    binary_mask = (predicted_mask> threshold).astype(np.uint8)  # Ensure 0-255 range
    # binary_mask_gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    # height, width = binary_mask.shape  # Lấy chiều cao và chiều rộng từ mảng pixel
    # image1 = np.reshape(binary_mask, (height, width)).astype(np.uint8)
    # out_mask = cv2.imshow(image1)

    # Save the segmented image
    # output_path = "testup/predicted_mask.png"
    # cv2.imwrite(output_path, binary_mask)

# Lưu ảnh binary_mask vào một thư mục
    output_folder = 'segmentation_results'
    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'binary_mask.png'
    mpimg.imsave(os.path.join(output_folder, output_filename), binary_mask, cmap='gray')

    
