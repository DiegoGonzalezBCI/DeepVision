import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2
import pickle
import glob

class Bildvorverarbeitung:
    def __init__(self, image_directory, excel_directory, target_height, target_width, x_offset, y_offset): 
        self.image_directory = image_directory
        self.excel_directory = excel_directory                                                             
        self.target_height   = target_height                                    
        self.target_width    = target_width
        self.x_offset        = x_offset
        self.y_offset        = y_offset
        self.images          = self.load_images_from_directory_call()

    def load_images_from_directory_call(self):
        images, diopt, vordiopt = self.load_images_from_directory()
        return images, diopt, vordiopt
    
    def load_images_from_directory(self):
        images = []  
        diopt = []  
        vordiopt = []  

        df = pd.read_excel(self.excel_directory) 
        title_exc = df['Title'].tolist()
        sample_exc = df['Sample'].tolist()
        diopt_pre = df['OPTIC OF - Pre VD PB'].tolist()
        diopt_post = df['OPTIC OF - Post VD PB'].tolist()

        number_of_images_in_folders = 0
        number_of_images_in_folders_and_excel = 0
        number_of_images_in_folders_and_excel_with_data = 0

        for directory in self.image_directory:
            directory_filenames = os.listdir(directory)
            for filename in directory_filenames:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    number_of_images_in_folders += 1
                    base_name, ext = os.path.splitext(filename)
                    parts = base_name.split("_")

                    title_img = parts[2][:6]
                    sample_img = parts[2][7:]

                    idx_title = [i for i, a in enumerate(title_exc) if a == title_img]
                    idx_sample   = [i for i, a in enumerate(sample_exc) if a == int(sample_img)]

                    if len(idx_title) != 0 and len(idx_sample) != 0:
                        if len(np.intersect1d(idx_title, idx_sample)) != 0:
                            number_of_images_in_folders_and_excel += 1
                            idx = np.intersect1d(idx_title, idx_sample)[0] 
                        else:
                            continue
                        if not(np.isnan(diopt_post[idx])) and not(np.isnan(diopt_pre[idx])):
                            number_of_images_in_folders_and_excel_with_data += 1
                            diopt_delta = diopt_post[idx] - diopt_pre[idx]
                            vordiopt_ = diopt_pre[idx]
                        else:
                            continue                    
                    else:
                        continue

                    images.append(cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE))
                    diopt.append(diopt_delta)
                    vordiopt.append(vordiopt_)

        print('Number of images in folders: ', number_of_images_in_folders)
        print('Number of images in folders and excel: ', number_of_images_in_folders_and_excel)
        print('Number of images in folders and excel with data: ', number_of_images_in_folders_and_excel_with_data)
        return images, diopt, vordiopt
    
    def crop_images(self, images):
        cropped_images = []
        for img in images:
            if img.shape != (self.target_height, self.target_width):
                original_height, original_width = img.shape[:2]
                left = ((original_width - self.target_width) // 2 ) + self.x_offset
                top = ((original_height - self.target_height) // 2 ) + self.y_offset
                right = left + self.target_width 
                bottom = top + self.target_height 
                cropped_images.append(img[top:bottom, left:right]) 

            else:
                cropped_images.append(img) 
        return cropped_images
    
def execute(dir, foldername1, foldername2, foldername3, saved_images_folder, excel):
    image_directory = [ dir + '\\' + foldername1, 
                        dir + '\\' + foldername2, 
                        dir + '\\' + foldername3 ] 
    excel_directory = excel
    image_processor = Bildvorverarbeitung(image_directory, excel_directory, target_height=4504, target_width=4504, x_offset=0, y_offset=0) 

    images   = image_processor.images[0]
    diopts   = image_processor.images[1]
    vordiopt = image_processor.images[2]

    images = image_processor.crop_images(images)

    directory = dir + '\\' + saved_images_folder + '\*'
    
    files = glob.glob(directory)

    for file in files:
        if os.path.exists(file):
            os.remove(file)

    file_number = 0
    for img in images:
        file_number += 1
        output_path = os.path.join(dir + '\\' + saved_images_folder, f"original_{file_number}.jpg")
        plt.imsave(output_path, img, cmap='gray', pil_kwargs={'compress_level': 0})

    with open("test", "wb") as fp:   
        pickle.dump(diopts, fp)

    with open("vordiopt", "wb") as fp:   
        pickle.dump(vordiopt, fp)
