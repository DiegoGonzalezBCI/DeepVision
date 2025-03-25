import cv2
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

def blackbox_BDP_func(dir, folder, filename, model_bubbles, model_differenz, model_mlp, diopter_before_bubble_removal, image_size, doplot):
    # Read the image
    img = cv2.imread(os.path.join(dir + '\\' + folder, filename), cv2.IMREAD_GRAYSCALE)
    img = [cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)]
    img = [img_ / 255 for img_ in img]
    img = [tf.cast(img_, dtype=tf.float32) for img_ in img]
    img = np.array(img)

    # Load models
    model_bubble = tf.keras.models.load_model(model_bubbles)
    model_differenz = tf.keras.models.load_model(model_differenz)
    model_mlp = tf.keras.models.load_model(model_mlp)

    # Predict bubble mask
    bubble_mask = model_bubble.predict(img)[0][:, :, 0]
    x_bubbles = np.sum(bubble_mask < 0.25)

    # Predict segment mask
    differenz_mask = model_differenz.predict(img)[0][:, :, 0]
    x_differenz = np.sum(differenz_mask < 0.25)

    # Prepare input for MLP
    X = np.array([[x_bubbles, x_differenz, diopter_before_bubble_removal]])

    # Predict final diopter
    prediction = model_mlp.predict(X)
    final_diopter = prediction[0][0]

    if doplot:
        # Plot results
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img[0], cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        inverted_image = 1 - bubble_mask
        axs[1].imshow(inverted_image, cmap='gray')
        axs[1].set_title(f"Bubble Mask - Pixel count: {x_bubbles}")
        axs[1].axis("off")

        inverted_image = 1 - differenz_mask
        axs[2].imshow(inverted_image, cmap='gray')
        axs[2].set_title(f"Differenz Mask - Pixel count: {x_differenz}")
        axs[2].axis("off")

        rounded_number = round(final_diopter, 2)
        fig.suptitle(f"Final Diopter is: {rounded_number:.2f}", fontsize=14)
        plt.tight_layout()
        plt.show()

    print('Predicted Final Diopter: ', prediction[0][0])

def blackbox_BP_func(dir, folder, filename, model_bubbles, model_mlp, diopter_before_bubble_removal, image_size, doplot):
    # Read the image
    img = cv2.imread(os.path.join(dir + '\\' + folder, filename), cv2.IMREAD_GRAYSCALE)
    img = [cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)]
    img = [img_ / 255 for img_ in img]
    img = [tf.cast(img_, dtype=tf.float32) for img_ in img]
    img = np.array(img)

    # Load models
    model_bubble = tf.keras.models.load_model(model_bubbles)
    model_mlp = tf.keras.models.load_model(model_mlp)

    # Predict bubble mask
    bubble_mask = model_bubble.predict(img)[0][:, :, 0]
    x_bubbles = np.sum(bubble_mask < 0.25)

    # Prepare input for MLP
    X = np.array([[x_bubbles, diopter_before_bubble_removal]])

    # Predict final diopter
    prediction = model_mlp.predict(X)
    final_diopter = prediction[0][0]

    if doplot:
        # Plot results
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(img[0], cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        inverted_image = 1 - bubble_mask
        axs[1].imshow(inverted_image, cmap='gray')
        axs[1].set_title(f"Bubble Mask - Pixel count: {x_bubbles}")
        axs[1].axis("off")

        rounded_number = round(final_diopter, 2)
        fig.suptitle(f"Final Diopter is: {rounded_number:.2f}", fontsize=14)
        plt.tight_layout()
        plt.show()

    print('Predicted Final Diopter: ', prediction[0][0])

def execute(dir, folder, filename, using_bubbles_and_differenz, model_bubbles, model_differenz, model_mlp, diopter_before_bubble_removal, image_size, doplot):
    if using_bubbles_and_differenz:
        blackbox_BDP_func(dir, folder, filename, model_bubbles, model_differenz, model_mlp, diopter_before_bubble_removal, image_size, doplot)
    elif not(using_bubbles_and_differenz):
        blackbox_BP_func(dir, folder, filename, model_bubbles, model_mlp, diopter_before_bubble_removal, image_size, doplot)