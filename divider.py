import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import cv2
import glob

def sort_and_batch_lists(nachdelta, list2, k):
    # Sort the first list
    sorted_list1 = sorted(nachdelta)
    
    # Initialize new lists
    nachdelta_1 = []
    nachdelta_2 = []
    list_1      = []
    list_2      = []
    
    # Iterate over batches of k elements
    for i in range(0, len(nachdelta), k):
        batch1 = sorted_list1[i:i+k]
        batch2 = [list2[nachdelta.index(x)] for x in batch1]
        
        # Split the batch into two halves
        half = len(batch1) // 2
        nachdelta_1.extend(batch1[:half])
        nachdelta_2.extend(batch1[half:])
        list_1.extend(batch2[:half])
        list_2.extend(batch2[half:])
    
    return nachdelta_1, nachdelta_2, list_1, list_2

def execute(dir, saved_images_folder, feature_to_divide, divided_images, divided_features, doplot):
    images_folder = dir + '\\' + saved_images_folder
    masks_folder = dir + '\\' + feature_to_divide

    images_files  = os.listdir(images_folder)
    masks_files  = os.listdir(masks_folder)

    r = re.compile(r'\d+')
    images_files.sort(key=lambda x: int(r.search(x).group()))
    masks_files.sort(key=lambda x: int(r.search(x).group()))

    with open("test", "rb") as fp:   
        diopts = pickle.load(fp)
    with open("vordiopt", "rb") as fp:   
        vordiopts = pickle.load(fp)

    original_y = diopts
    y = list(filter(lambda x: -10 < x < 0, original_y))
    original_y = -np.array(original_y)
    y = -np.array(y)
    preserved_img = [1 if x in y else 0 for x in original_y]

    images    = []
    masks     = []
    delta     = []
    vordelta  = []
    nachdelta = []
    for (image_file, mask_file, delta_, vordelta_, preserved) in zip(images_files, masks_files, original_y, vordiopts, preserved_img):
        if preserved:
            image = cv2.imread(os.path.join(images_folder, image_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            images.append(image)
            masks.append(mask)
            delta.append(delta_)
            vordelta.append(vordelta_)
            nachdelta.append(vordelta_ - delta_)

    k                                                   = 2
    _, _, images_1, images_2    = sort_and_batch_lists(nachdelta, images, k)
    images                                              = images_1 + images_2
    _, _, masks_1, masks_2    = sort_and_batch_lists(nachdelta, masks, k)
    masks                                              = masks_1 + masks_2
    _, _, delta_1, delta_2      = sort_and_batch_lists(nachdelta, delta, k)
    delta                                               = delta_1 + delta_2
    nachdelta_1, nachdelta_2, vordelta_1, vordelta_2    = sort_and_batch_lists(nachdelta, vordelta, k)
    vordelta                                            = vordelta_1 + vordelta_2
    nachdelta                                           = nachdelta_1 + nachdelta_2

    directory = dir + '\\' + divided_images + '\*'
    files = glob.glob(directory)
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    directory = dir + '\\' + divided_features + '\*'
    files = glob.glob(directory)
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    file_number = 0
    for img in images:
        file_number += 1
        output_path = os.path.join(dir + '\\' + divided_images, f"original_{file_number}.jpg")
        plt.imsave(output_path, img, cmap='gray', pil_kwargs={'compress_level': 0})
    file_number = 0
    for img in masks:
        file_number += 1
        output_path = os.path.join(dir + '\\' + divided_features, f"mask_{file_number}.jpg")
        plt.imsave(output_path, img, cmap='gray', pil_kwargs={'compress_level': 0})

    with open("testR", "wb") as fp:   
        pickle.dump(delta, fp)
    with open("vordioptR", "wb") as fp:   
        pickle.dump(vordelta, fp)
    with open("nachdioptR", "wb") as fp:   
        pickle.dump(nachdelta, fp)

    if doplot:
        plt.hist(nachdelta, bins=np.arange(10, 35.01, 0.1), edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Zieldiopter Distribution")
        plt.grid(True)
        plt.show()

        plt.hist(nachdelta[: len(nachdelta) // 2], bins=np.arange(10, 35.01, 0.1), edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Zieldiopter Distribution (First division)")
        plt.grid(True)
        plt.show()

        plt.hist(nachdelta[ len(nachdelta) // 2 :], bins=np.arange(10, 35.01, 0.1), edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Zieldiopter Distribution (Second division)")
        plt.grid(True)
        plt.show()