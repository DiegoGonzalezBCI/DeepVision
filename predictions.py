import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import re
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error

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
            fig, axs = plt.subplots(2, 3, figsize=(12, 15))

            for i in range(2):
                for j in range(3):
                    index = i * 3 + j
                    if index < len(self.images):
                        axs[i, j].imshow(self.images[index], cmap='gray')
                        axs[i, j].axis("off") 

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.show()

def shift_image_and_mask(image, max_shift, image_size):
    # Generate random shift values
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    # Shift the image
    shifted_image = np.roll(image, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)

    # Crop 
    shifted_image = shifted_image[:image_size, :image_size]
    return shifted_image

def plot_and_evaluate(history, Y_test, predictions, doplot):
    if doplot:
        # Plot training and validation loss curves
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.show()

        # Plot predictions vs. true values
        plt.figure()
        plt.scatter(Y_test, predictions, alpha=0.5)
        plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='r', linestyle='--', label='Ideal Line')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. True Values')
        plt.show()

    # Calculate evaluation metrics
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    rmse = mean_squared_error(Y_test, predictions, squared=False)
    mae = mean_absolute_error(Y_test, predictions)
    maxe = max_error(Y_test, predictions)

    # Return evaluation metrics
    return mse, r2, rmse, mae, maxe

def bubbles_and_differenz(dir, saved_images_folder, using_divided_images, model_bubbles, model_differenz, image_size, number_of_times_images_will_be_augmented, model1, model2, model3, model4, model5, doplot):
    model_bubble           = tf.keras.models.load_model(model_bubbles)
    model_segmen           = tf.keras.models.load_model(model_differenz)

    original_folder        = dir + '\\' + saved_images_folder
    original_files         = os.listdir(original_folder)
    r = re.compile(r'\d+')
    original_files.sort(key=lambda x: int(r.search(x).group()))
    new_height = image_size
    new_width = image_size

    images = []
    yd = []
    yv  = []

    if not(using_divided_images):
        with open("test", "rb") as fp:   
            diopts = pickle.load(fp)
        with open("vordiopt", "rb") as fp:   
            vordiopt = pickle.load(fp)

        original_y = diopts
        y = list(filter(lambda x: -10 < x < 0, original_y))
        original_y = -np.array(original_y)
        vordiopt   = np.array(vordiopt)
        y = -np.array(y)
        preserved_img = [1 if x in y else 0 for x in original_y]

        for (original_file, y_delta, y_vor, preserved) in zip(original_files, original_y, vordiopt, preserved_img):
            if preserved:
                image = cv2.imread(os.path.join(original_folder, original_file), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                images.append(image)
                yd.append(y_delta)
                yv.append(y_vor)

    elif using_divided_images:
        with open("testR", "rb") as fp:   
            diopts = pickle.load(fp)
        with open("vordioptR", "rb") as fp:   
            vordiopt = pickle.load(fp)

        for (original_file, y_delta, y_vor) in zip(original_files, diopts, vordiopt):
            image = cv2.imread(os.path.join(original_folder, original_file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            yd.append(y_delta)
            yv.append(y_vor)

        half = len(images) // 2
        images = images[half:]
        yd  = yd[half:]
        yv = yv[half:]

    indices = list(range(len(images)))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    yd     = [yd[i] for i in indices]
    yv     = [yv[i] for i in indices]

    total_samples = len(images)
    num_folds = 5
    fold_size = total_samples // (2 * num_folds)
    model_metrics_list = [] 

    for j in range(num_folds):

        start_val = 2 * j * fold_size
        end_val = ((2 * j) + 1) * fold_size
        start_test = end_val
        end_test = ((2 * j) + 2) * fold_size if j < num_folds - 1 else total_samples

        images_train = images[:start_val] + images[end_test:]
        images_val = images[start_val:end_val]
        images_test = images[start_test:end_test]

        yd_train = yd[:start_val] + yd[end_test:]
        yd_val = yd[start_val:end_val]
        yd_test = yd[start_test:end_test]

        yv_train = yv[:start_val] + yv[end_test:]
        yv_val = yv[start_val:end_val]
        yv_test = yv[start_test:end_test]

        for k in range(3):
            if k == 0:
                img    = images_train
                ydelta = yd_train
                yvordi = yv_train
                do_DA  = 1
            elif k == 1:
                img    = images_val
                ydelta = yd_val
                yvordi = yv_val
                do_DA  = 1
            elif k == 2:
                img    = images_test
                ydelta = yd_test
                yvordi = yv_test
                do_DA = 0

            images_aug, ydelta_aug, yvordi_aug = [], [], []
            for (image, yd_, yv_) in zip(img, ydelta, yvordi):
                if do_DA:
                    for _ in range(number_of_times_images_will_be_augmented):  
                        angle = random.uniform(0, 360)
                        M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
                        rotated_image = cv2.warpAffine(image, M, (new_width, new_height))
                        shifted_image = shift_image_and_mask(rotated_image, np.floor(image_size/12.5), image_size)
                        images_aug.append(shifted_image)
                        ydelta_aug.append(yd_)
                        yvordi_aug.append(yv_)
                images_aug.append(image)
                ydelta_aug.append(yd_)
                yvordi_aug.append(yv_)
        
            num_images = len(images_aug)
            index_list = list(range(num_images))
            random.shuffle(index_list)
            images_aug = [images_aug[i] for i in index_list]
            ydelta_aug = [ydelta_aug[i] for i in index_list]
            yvordi_aug = [yvordi_aug[i] for i in index_list]

            images_aug = [image_aug  / 255 for image_aug  in images_aug]
            images_aug = [tf.cast(image_aug, dtype=tf.float32) for image_aug in images_aug]
            images_aug = np.array(images_aug)

            image_bubble = model_bubble.predict(images_aug)
            image_segmen = model_segmen.predict(images_aug)

            x_bubble, x_segmen = [], []

            for bubble_mask, segmen_mask in zip(image_bubble, image_segmen):
                bubble_mask = bubble_mask[:, :, 0]
                segmen_mask = segmen_mask[:, :, 0]
                num_pixels_bubbles = np.sum(bubble_mask < 0.25) 
                num_pixels_segmen = np.sum(segmen_mask < 0.25) 
                x_bubble.append(num_pixels_bubbles)
                x_segmen.append(num_pixels_segmen)

            if k == 0:
                x_train_bubbles = x_bubble
                x_train_segmen  = x_segmen
                x_train_vordio  = yvordi_aug
                y_train_delta   = ydelta_aug
            elif k == 1:
                x_val_bubbles = x_bubble
                x_val_segmen  = x_segmen
                x_val_vordio  = yvordi_aug
                y_val_delta   = ydelta_aug
            elif k == 2:
                x_test_bubbles = x_bubble
                x_test_segmen  = x_segmen
                x_test_vordio  = yvordi_aug
                y_test_delta   = ydelta_aug

        Y_train = np.array(x_train_vordio) - np.array(y_train_delta)
        Y_val   = np.array(x_val_vordio) - np.array(y_val_delta)
        Y_test   = np.array(x_test_vordio) - np.array(y_test_delta)

        X_train  = np.array([x_train_bubbles, x_train_segmen, x_train_vordio]).T
        X_val    = np.array([x_val_bubbles, x_val_segmen, x_val_vordio]).T
        X_test   = np.array([x_test_bubbles, x_test_segmen, x_test_vordio]).T

        if j == 0:
            model_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_1.summary()
            history = model_1.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val)) 
            predictions = model_1.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 1:
            model_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_2.summary()
            history = model_2.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_2.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 2:
            model_3 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_3.summary()
            history = model_3.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_3.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 3:
            model_4 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_4.summary()
            history = model_4.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_4.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 4:
            model_5 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_5.summary()
            history = model_5.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_5.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

    averaged_metrics = np.mean(model_metrics_list, axis=0)
    metrics = ["MSE", "R2", "RMSE", "MAE", "MAXE"]
    print(f"{'Model':<10}{'MSE':<10}{'R2':<10}{'RMSE':<10}{'MAE':<10}{'MAXE':<10}")
    print("-" * 60)
    for i, model in enumerate(model_metrics_list, start=1):
        print(f"Model {i:<8}", end="")
        for metric in model:
            print(f"{metric:<10.2f}", end="")
        print()
    print("-" * 60)

    print(f"Averaged Metrics through a 5-fold Cross Validation:")
    print(f"MSE: {averaged_metrics[0]}")
    print(f"R2: {averaged_metrics[1]}")
    print(f"RMSE: {averaged_metrics[2]}")
    print(f"MAE: {averaged_metrics[3]}")
    print(f"Max Error: {averaged_metrics[4]}")

    if doplot:
        plt.hist(list(Y_train) + list(Y_val) + list(Y_test), bins=np.arange(10, 35.01, 0.1), edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Diopter-Change Distribution")
        plt.grid(True)
        plt.show()

    model_1.save(model1)
    model_2.save(model2)
    model_3.save(model3)
    model_4.save(model4)
    model_5.save(model5)

def just_bubbles(dir, saved_images_folder, using_divided_images, model_bubbles, image_size, number_of_times_images_will_be_augmented, model1, model2, model3, model4, model5, doplot):
    model_bubble           = tf.keras.models.load_model(model_bubbles)

    original_folder        = dir + '\\' + saved_images_folder
    original_files         = os.listdir(original_folder)
    r = re.compile(r'\d+')
    original_files.sort(key=lambda x: int(r.search(x).group()))
    new_height = image_size
    new_width = image_size

    images = []
    yd = []
    yv  = []

    if not(using_divided_images):
        with open("test", "rb") as fp:   
            diopts = pickle.load(fp)
        with open("vordiopt", "rb") as fp:   
            vordiopt = pickle.load(fp)

        original_y = diopts
        y = list(filter(lambda x: -10 < x < 0, original_y))
        original_y = -np.array(original_y)
        vordiopt   = np.array(vordiopt)
        y = -np.array(y)
        preserved_img = [1 if x in y else 0 for x in original_y]

        for (original_file, y_delta, y_vor, preserved) in zip(original_files, original_y, vordiopt, preserved_img):
            if preserved:
                image = cv2.imread(os.path.join(original_folder, original_file), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                images.append(image)
                yd.append(y_delta)
                yv.append(y_vor)

    elif using_divided_images:
        with open("testR", "rb") as fp:   
            diopts = pickle.load(fp)
        with open("vordioptR", "rb") as fp:   
            vordiopt = pickle.load(fp)

        for (original_file, y_delta, y_vor) in zip(original_files, diopts, vordiopt):
            image = cv2.imread(os.path.join(original_folder, original_file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            yd.append(y_delta)
            yv.append(y_vor)

        half = len(images) // 2
        images = images[half:]
        yd  = yd[half:]
        yv = yv[half:]

    indices = list(range(len(images)))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    yd     = [yd[i] for i in indices]
    yv     = [yv[i] for i in indices]

    total_samples = len(images)
    num_folds = 5
    fold_size = total_samples // (2 * num_folds)
    model_metrics_list = [] 

    for j in range(num_folds):

        start_val = 2 * j * fold_size
        end_val = ((2 * j) + 1) * fold_size
        start_test = end_val
        end_test = ((2 * j) + 2) * fold_size if j < num_folds - 1 else total_samples

        images_train = images[:start_val] + images[end_test:]
        images_val = images[start_val:end_val]
        images_test = images[start_test:end_test]

        yd_train = yd[:start_val] + yd[end_test:]
        yd_val = yd[start_val:end_val]
        yd_test = yd[start_test:end_test]

        yv_train = yv[:start_val] + yv[end_test:]
        yv_val = yv[start_val:end_val]
        yv_test = yv[start_test:end_test]

        for k in range(3):
            if k == 0:
                img    = images_train
                ydelta = yd_train
                yvordi = yv_train
                do_DA  = 1
            elif k == 1:
                img    = images_val
                ydelta = yd_val
                yvordi = yv_val
                do_DA  = 1
            elif k == 2:
                img    = images_test
                ydelta = yd_test
                yvordi = yv_test
                do_DA = 0

            images_aug, ydelta_aug, yvordi_aug = [], [], []
            for (image, yd_, yv_) in zip(img, ydelta, yvordi):
                if do_DA:
                    for _ in range(number_of_times_images_will_be_augmented):  
                        angle = random.uniform(0, 360)
                        M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
                        rotated_image = cv2.warpAffine(image, M, (new_width, new_height))
                        shifted_image = shift_image_and_mask(rotated_image, np.floor(image_size/12.5), image_size)
                        images_aug.append(shifted_image)
                        ydelta_aug.append(yd_)
                        yvordi_aug.append(yv_)
                images_aug.append(image)
                ydelta_aug.append(yd_)
                yvordi_aug.append(yv_)
        
            num_images = len(images_aug)
            index_list = list(range(num_images))
            random.shuffle(index_list)
            images_aug = [images_aug[i] for i in index_list]
            ydelta_aug = [ydelta_aug[i] for i in index_list]
            yvordi_aug = [yvordi_aug[i] for i in index_list]

            images_aug = [image_aug  / 255 for image_aug  in images_aug]
            images_aug = [tf.cast(image_aug, dtype=tf.float32) for image_aug in images_aug]
            images_aug = np.array(images_aug)

            image_bubble = model_bubble.predict(images_aug)

            x_bubble = []

            for bubble_mask in image_bubble:
                bubble_mask = bubble_mask[:, :, 0]
                num_pixels_bubbles = np.sum(bubble_mask < 0.25) 
                x_bubble.append(num_pixels_bubbles)

            if k == 0:
                x_train_bubbles = x_bubble
                x_train_vordio  = yvordi_aug
                y_train_delta   = ydelta_aug
            elif k == 1:
                x_val_bubbles = x_bubble
                x_val_vordio  = yvordi_aug
                y_val_delta   = ydelta_aug
            elif k == 2:
                x_test_bubbles = x_bubble
                x_test_vordio  = yvordi_aug
                y_test_delta   = ydelta_aug

        Y_train = np.array(x_train_vordio) - np.array(y_train_delta)
        Y_val   = np.array(x_val_vordio) - np.array(y_val_delta)
        Y_test   = np.array(x_test_vordio) - np.array(y_test_delta)

        X_train  = np.array([x_train_bubbles, x_train_vordio]).T
        X_val    = np.array([x_val_bubbles, x_val_vordio]).T
        X_test   = np.array([x_test_bubbles, x_test_vordio]).T

        if j == 0:
            model_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_1.summary()
            history = model_1.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val)) 
            predictions = model_1.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 1:
            model_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_2.summary()
            history = model_2.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_2.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 2:
            model_3 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_3.summary()
            history = model_3.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_3.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 3:
            model_4 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_4.summary()
            history = model_4.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_4.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

        elif j == 4:
            model_5 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])
            model_5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
            model_5.summary()
            history = model_5.fit(X_train, Y_train, epochs=300, batch_size=16, validation_data=(X_val, Y_val))
            predictions = model_5.predict(X_test)
            mse, r2, rmse, mae, maxe = plot_and_evaluate(history, Y_test, predictions, doplot)

            model_metrics_list.append([mse, r2, rmse, mae, maxe])

    averaged_metrics = np.mean(model_metrics_list, axis=0)
    metrics = ["MSE", "R2", "RMSE", "MAE", "MAXE"]
    print(f"{'Model':<10}{'MSE':<10}{'R2':<10}{'RMSE':<10}{'MAE':<10}{'MAXE':<10}")
    print("-" * 60)
    for i, model in enumerate(model_metrics_list, start=1):
        print(f"Model {i:<8}", end="")
        for metric in model:
            print(f"{metric:<10.2f}", end="")
        print()
    print("-" * 60)

    print(f"Averaged Metrics through a 5-fold Cross Validation:")
    print(f"MSE: {averaged_metrics[0]}")
    print(f"R2: {averaged_metrics[1]}")
    print(f"RMSE: {averaged_metrics[2]}")
    print(f"MAE: {averaged_metrics[3]}")
    print(f"Max Error: {averaged_metrics[4]}")

    if doplot:
        plt.hist(list(Y_train) + list(Y_val) + list(Y_test), bins=np.arange(10, 35.01, 0.1), edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Diopter-Change Distribution")
        plt.grid(True)
        plt.show()

    model_1.save(model1)
    model_2.save(model2)
    model_3.save(model3)
    model_4.save(model4)
    model_5.save(model5)

def execute(dir, saved_images_folder, using_bubbles_and_differenz, using_divided_images, model_bubbles, model_differenz, image_size, number_of_times_images_will_be_augmented, model1, model2, model3, model4, model5, doplot):
    if using_bubbles_and_differenz:
        bubbles_and_differenz(dir, saved_images_folder, using_divided_images, model_bubbles, model_differenz, image_size, number_of_times_images_will_be_augmented, model1, model2, model3, model4, model5, doplot)
    elif not(using_bubbles_and_differenz):
        just_bubbles(dir, saved_images_folder, using_divided_images, model_bubbles, image_size, number_of_times_images_will_be_augmented, model1, model2, model3, model4, model5, doplot)