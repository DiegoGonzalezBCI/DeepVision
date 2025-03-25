import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
from keras import layers
import pickle
import tensorflow as tf
import random
import re

class BildPlotter:
    def __init__(self, images):
        self.images = images

    def plot_image(self, option):
        if option == 1:
            fig, ax = plt.subplots()
            ax.imshow(self.images, cmap='gray')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title('Sample Image')
            plt.show()
        elif option == 2:
            fig, axs = plt.subplots(2, 3, figsize=(15, 15))

            for i in range(2):
                for j in range(3):
                    index = i * 3 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

class NeuralNet:
    # Define a custom U-Net model
    def custom_unet_model(self, input_shape):
        
        inputs = tf.keras.Input(shape=input_shape)

        # Here goes the UNet defined by the Functional API Blog
        def double_conv_block(x, n_filters):

            # Conv2D then ReLU activation
            x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
            # Conv2D then ReLU activation
            x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

            return x
        def downsample_block(x, n_filters):
            f = double_conv_block(x, n_filters)
            p = layers.MaxPool2D(2)(f)
            p = layers.Dropout(0.3)(p)

            return f, p
        def upsample_block(x, conv_features, n_filters):
            # upsample
            x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
            # concatenate
            x = layers.concatenate([x, conv_features])
            # dropout
            x = layers.Dropout(0.3)(x)
            # Conv2D twice with ReLU activation
            x = double_conv_block(x, n_filters)

            return x
            
        inputs = layers.Input(shape=input_shape)
        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = downsample_block(inputs, 64)
        # 2 - downsample
        f2, p2 = downsample_block(p1, 128)
        # 3 - downsample
        f3, p3 = downsample_block(p2, 256)
        # 4 - downsample
        f4, p4 = downsample_block(p3, 512)

        # 5 - bottleneck
        bottleneck = double_conv_block(p4, 1024)

        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = upsample_block(bottleneck, f4, 512)
        # 7 - upsample
        u7 = upsample_block(u6, f3, 256)
        # 8 - upsample
        u8 = upsample_block(u7, f2, 128)
        # 9 - upsample
        u9 = upsample_block(u8, f1, 64)

        # outputs
        outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9) # 3 and softmax with SparseCategoricalCrossentropy or 1 and sigmoid with BinaryCrossentropy

        model = tf.keras.Model(inputs, outputs, name="U-Net")
        return model
    
def shift_image_and_mask(image, mask, max_shift, image_size):
    # Generate random shift values
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    # Shift the image
    shifted_image = np.roll(image, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)

    # Shift the mask
    shifted_mask = np.roll(mask, shift_x, axis=1)
    shifted_mask = np.roll(shifted_mask, shift_y, axis=0)

    # Crop
    shifted_image = shifted_image[:image_size, :image_size]
    shifted_mask = shifted_mask[:image_size, :image_size]

    return shifted_image, shifted_mask

def execute(dir, saved_images_folder, masks_to_use, using_divided_images, image_size, number_of_times_images_will_be_augmented, name_model_to_save, doplot):
    images_folder = dir + '\\' + saved_images_folder
    masks_folder = dir + '\\' + masks_to_use

    images_files  = os.listdir(images_folder)
    masks_files = os.listdir(masks_folder)

    r = re.compile(r'\d+')
    images_files.sort(key=lambda x: int(r.search(x).group()))
    masks_files.sort(key=lambda x: int(r.search(x).group()))

    new_height = image_size
    new_width = image_size

    with open("nachdioptR", "rb") as fp:   
        nachdelta = pickle.load(fp)

    images = []
    masks = []
    ydelta = []

    if not(using_divided_images):
        with open("test", "rb") as fp:   
            diopts = pickle.load(fp)
    
        original_y = diopts
        y = list(filter(lambda x: -10 < x < 0, original_y))
        original_y = -np.array(original_y)
        y = -np.array(y)
        preserved_img = [1 if x in y else 0 for x in original_y]

        for (image_file, mask_file, y_delta, preserved) in zip(images_files, masks_files, original_y, preserved_img):
            if preserved:
                image = cv2.imread(os.path.join(images_folder, image_file), cv2.IMREAD_GRAYSCALE)
                mask  = cv2.imread(os.path.join(masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                mask  = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                images.append(image)
                masks.append(mask)
                ydelta.append(y_delta)
    elif using_divided_images:
        with open("testR", "rb") as fp:   
            diopts = pickle.load(fp)

        for (image_file, mask_file, y_delta) in zip(images_files, masks_files, diopts):
            image = cv2.imread(os.path.join(images_folder, image_file), cv2.IMREAD_GRAYSCALE)
            mask  = cv2.imread(os.path.join(masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            masks.append(mask)
            ydelta.append(y_delta)

        half = len(images) // 2
        images = images[:half]
        masks  = masks[:half]
        ydelta = ydelta[:half]

    indices = list(range(len(images)))
    random.shuffle(indices)
    images  = [images[i] for i in indices]
    masks   = [masks[i]  for i in indices]
    ydelta  = [ydelta[i] for i in indices]

    images_size = int(0.8 * len(images))
    images_train, images_temp = images[:images_size], images[images_size:]
    images_size = int(0.75 * len(images_temp))
    images_val, images_test = images_temp[:images_size], images_temp[images_size:]

    masks_size = int(0.8 * len(masks))
    masks_train, masks_temp = masks[:masks_size], masks[masks_size:]
    masks_size = int(0.75 * len(masks_temp))
    masks_val, masks_test = masks_temp[:masks_size], masks_temp[masks_size:]

    ydelta_size = int(0.8 * len(ydelta))
    ydelta_train, ydelta_temp = ydelta[:ydelta_size], ydelta[ydelta_size:]
    ydelta_size = int(0.75 * len(ydelta_temp))
    ydelta_val, ydelta_test = ydelta_temp[:ydelta_size], ydelta_temp[ydelta_size:]

    for k in range(3):
        if k == 0:
            images = images_train
            masks = masks_train
            ydelta = ydelta_train
            do_DA = 1
        elif k == 1:
            images = images_val
            masks = masks_val
            ydelta = ydelta_val
            do_DA = 1
        elif k == 2:
            images = images_test
            masks = masks_test
            ydelta = ydelta_test
            do_DA = 0

        images_aug, masks_aug, ydelta_aug = [], [], []
        for (image, mask, ydel) in zip(images, masks, ydelta):
            if do_DA:
                for _ in range(number_of_times_images_will_be_augmented): 

                    angle = random.uniform(0, 360)
                    M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
                    rotated_image = cv2.warpAffine(image, M, (new_width, new_height))
                    rotated_mask = cv2.warpAffine(mask, M, (new_width, new_height))
                    shifted_image, shifted_mask = shift_image_and_mask(rotated_image, rotated_mask, np.floor(image_size/12.5), image_size)            
                    images_aug.append(shifted_image)
                    masks_aug.append(shifted_mask)
                    ydelta_aug.append(ydel)
            images_aug.append(image)
            masks_aug.append(mask)
            ydelta_aug.append(ydel)

        num_images = len(images_aug)
        index_list = list(range(num_images))
        random.shuffle(index_list)
        images_aug = [images_aug[i] for i in index_list]
        masks_aug  = [masks_aug[i] for i in index_list]
        ydelta_aug = [ydelta_aug[i] for i in index_list]

        images_aug = [image_aug  / 255 for image_aug  in images_aug]
        images_aug = [tf.cast(image_aug, dtype=tf.float32) for image_aug in images_aug]
        images_aug = np.array(images_aug)

        masks_aug = [mask_aug  / 255 for mask_aug  in masks_aug]
        masks_aug = [tf.cast(mask_aug, dtype=tf.float32) for mask_aug in masks_aug]
        masks_aug = np.array(masks_aug)

        if k == 0:
            x_train  = images_aug
            y_train  = masks_aug
            yd_train = ydelta_aug
        elif k == 1:
            x_val    = images_aug
            y_val    = masks_aug
            yd_val   = ydelta_aug
        elif k == 2:
            x_test   = images_aug
            y_test   = masks_aug
            yd_test  = ydelta_aug

############################### - Train the Neural Network - #################################

    neural_net = NeuralNet()
    input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1) 
    model = neural_net.custom_unet_model(input_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])   

    model.summary()

    history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_val, y_val))

    model.save(name_model_to_save)

    if doplot:
        plt.hist(nachdelta, bins=np.arange(10, 35.01, 0.1), edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Diopter-Change Distribution")
        plt.grid(True)
        plt.show()

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')  
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.show()

        # Predict masks for test images
        predicted_masks = model.predict(x_test)

        num_images = 6
        rows, cols = 2, 3

        fig, axs = plt.subplots(rows, cols, figsize=(15, 8))

        x_test = [ (img - img.min()) / (img.max()- img.min()) for img in x_test]

        for i in range(num_images):
            row_idx = i // cols
            col_idx = i % cols

            # Combine original image and predicted mask using bitwise AND
            combined_image = cv2.bitwise_and(x_test[i], x_test[i], mask=(predicted_masks[i][:,:,0] < 0.25).astype(np.uint8))

            #axs[row_idx, col_idx].imshow(overlay_image, cmap="gray")
            axs[row_idx, col_idx].imshow(combined_image, cmap="gray")
            axs[row_idx, col_idx].set_title(f"Image {i+1}")
            axs[row_idx, col_idx].axis("off")

        plt.tight_layout()
        plt.show()