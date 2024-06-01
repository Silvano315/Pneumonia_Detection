import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image and convert to array using OpenCV with appropriate preprocessing.
def get_img_array(img_path, size):

    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)

    return img


# Generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Save and display Grad-CAM image with original image
def save_and_display_gradcam(img_path, heatmap, cam_path="static/cam.jpg", alpha=0.4):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    parts = img_path.split('/')
    filename = parts[-1].split('.')[0]
    cv2.imwrite('grad_cam_images/' + filename + cam_path, superimposed_img)
    
    superimposed_img=superimposed_img/ 255.0
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(superimposed_img[...,::-1])
    axes[0].axis('off')

    axes[1].imshow(img[...,::-1]) 
    axes[1].axis('off')
    plt.show()