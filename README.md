# DeepVision
The DeepVision Project uses Python, artificial intelligence, and computer vision techniques to develop an accurate model for predicting the diopter of Paris IOLs. By analyzing optical data, the model enhances IOL selection and reduces post-operative refractive errors, improving patient outcomes.

# Introduction 
    This project aims to develop an accurate prediction model for determining the diopter of Paris IOLs. 
    Our goal is to enhance patient outcomes by ensuring precise IOL selection and minimizing post-operative refractive errors.
    For further information please visit the folder DiopterPrediction under AzureDevOps GitHub.
    More information about technical specifications of each program can be found in a PowerPoint presentation called Dioptre Reduzierung.  

# Getting Started
1.	Installation process
    Ensure you have Python 3.0 or above installed on your system. If not, download and install it from the official Python website.
    Verify your installation by running python --version in your terminal.
2.	Software dependencies
    Install all needed python libraries using the latest release of pip. 
3.  Python Programs
    Used Python programs inclue: (1): Store_data.py, (2): Manual_masking.py, (3): UNet_DA_FV.py, (4): Multilayer_Perceptron_Test_DA_FV_CV_BS.py, (5): blackbox.py
    Programs 1 to 4 are used when feeding the models with more data to enhnace their performance.
    Program 5 is used just to call those trained models (3 in total) and perform diopter prediction. 

# Build and Test
- Testing Existing Models
    - Quick Testing with “blackbox.py”:
    - If you want to quickly test the existing machine learning models, follow these steps:
        - Run the blackbox.py script.
        - Input the image directory and filename.
        - Provide the corresponding diopter value.
    - The script will predict the diopter based on the pre-trained models.

- Enhancing Models with More Data
    - Semantic Segmentation and Manual Masking:
    - If you have additional data and want to improve the Semantic Segmentation models:
        
        - Collect more labeled images of Paris IOLs.
        - This data new collected data must follow the same naming as the files used so far. 
        - Naming style: 2023.09.11_13.33.11.548240_LA-040-003_Image
        - "LA-040-003" is the important part of the name, where "LA-040" is an excel-colum-key and "003" another excel-colum-key. (2-colum-keys per image)
        - "Store_data.py" will open the requested directories and open the excel file with the names and diopter value. 
        - When the image's colum keys meet the ones specified in the excel, and there exist values for the diopter before and after the bubble removal, the image is stored in a folder which is usually named "original" and the diopter-difference and diopter-before-bubble-removal are stored in "test" and "vordiopt" respectively. 
            
        - The next step is to run "Manual_Masking.py" which is a python program to do manual masking and enhnace the semantic segmentation models. 
        - The program opens the "original" folder and show all images to the user one by one. 
        - The user is expected to click on the image various times forming a path-like curve that encapsulates the desired feature to be extracted. 
        - We reffer to "Bubbles" as the feature which exhibits a bubble-like form in the IOL.
        - We reffer to "Differenz" as the feature which exhibits those perimetric areas of the IOL which were left out and are empty. 
        - We reffer to "Volume" as the feature which exhibits the total area occupied by the IOL. (This feature is not important and the user should not mask it).

        - The next step is to run "UNet_DA_FV.py" which is a python program to train the semantic segmentation models. 
        - As of 06.2024 the best hyperparameters to train the Bubble-Masking model and the Differenz-Masking models are:
        - Epochs: 50, Image size: 256x256, Batch size: 16, Rotation and Shifting Data Augmentation of x11 every original image (goal is to have at least 1000 images for training).
        - The user gives a generical name to save it and later be further utilized by the Multilayer Perceptron stage. 
        - To train the Bubbles-Differenz-Prediopt Multilayer Perceptron model, at least 2 times must be runned the Semantic Segmentation model (one for Bubbles' feature and another for Differenz' feature).

        - The next step is to run the "Multilayer_perceptron_Test_DA_FV_CV_BS.py" which is a python program to train the multilayer perceptron model in a 5-fold Cross Validation fashion, ensuring the reliance on the given metrics. 
        - As of 06.2024 the best hyperparameters to train the model are:
        - Epochs: 300, Image size: 256x256, Batch size: 16, Rotation and Shifting Data Augmentation of x11 every original image, Usage of Bubbles-Differenz-Prediopt*, 3-Layered net. 
        - *Prediopt is the diopter before bubble removal. 
        - The program uses the Bubble-model and Differenz-model to generate automatically the masks of the images within the specified directory, count the pixels, and train a multilayer perceptron model whose output is the final diopter after bubble removal. 
        - A 5-fold cross validation is recommended to assess in a reliable way the model's performance. 
        - Each sub-trained MLP-model has their own variable for it to be stored by choice by the user. 

    - If you prefer not to manually mask images:
        - Re-training the Multilayer Perceptron (MLP)
        - The Semantic Segmentation models should perform quite well with data whose features are alike to those used while training the model. 
        - Just include the additional images in the requested folder of this program and perform again the cross validation to obtain results reliable enough. 

    - After performing either model's update, the models specified in "blackbox.py" should be also updated. 

# Contribute 
All python files, as well as KERAS pre-trained models, exist in an AzureDevOps repository called "Diopter Prediction". 
