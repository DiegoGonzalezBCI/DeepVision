import save_data
import manual_masking
import divider
import automatic_masking
import predictions
import blackbox

# The first part of the program runs blackbox to test pre-trained model's performance over new data.
# The second part of the program runs the training chain used to create such models. 
# Run each subpart of the training chain independently and in order from 1 to 5 (subpart 3 can be skipped). 

# Change the following line to use another directory:
dir = r'C:\Users\SANCHDI2\OneDrive - Alcon\GitHub\dioptre_reduzierung'
# *All needed folders must be created before the program runs*

f1  = int(input('1: Test pre-trained models\n2: Train again with more data\n'))

if f1 == 1:
    blackbox.execute( 
            dir = dir,
            folder='images',
            filename='image_1.jpg',
            using_bubbles_and_differenz=False, # if True model_bubbles and model_differenz must be specified, otherwise only model_bubbles must be specified and model_differenz can be left as an empty string
            model_bubbles='UNet_Bubbles_size_128_DAS_DE_RED_epoche_50.keras',
            model_differenz='',
            model_mlp='MLP_BP_DA_11_DE_ZD_RED_3L_UNet_Voll_try_3_CrossValidation_epoche_150.keras',
            diopter_before_bubble_removal=25,
            image_size=128,
            doplot=True
            )
elif f1 == 2:
    f2 = int(input('1: Store new data\n2: Do manual masking\n3: Split data\n4: Train the Semantic Segmentation model\n5: Train the Multilayer Perceptron model\n'))
    if f2 == 1:
        save_data.execute(
            dir=dir,
            foldername1='Labeled1_18_4', # foldername1, 2 and 3 are the folders where the IOL images are saved
            foldername2='Labeled2_18_4',
            foldername3='New_erzeugte_Blasen_after',
            saved_images_folder='original', # This folder is where the images that are going to be used for the model's training are saved (notice that not all images saved in the past 3 folders are usable for model's trainning!)
            excel='new_diopters.xlsx'
            )
    elif f2 == 2:
        manual_masking.execute(
            dir=dir,
            saved_images_folder='original', # This folder is where the images that are going to be used for the model's training are saved
            feature_to_capture='volumen', # This folder is where the masks of the specified feature are saved
            start_from=132 # If the user want to start masking from a specific image (notice that images are named from 1 to N) (notice that first image correspond to 0 index)
        )
    elif f2 == 3:
        divider.execute( # Notice that by halving the total data a performance's decrease of the model is to be expected. Don't do it when few data is available. Don't do it if confidence about segmentation's model is high.
            dir=dir,
            saved_images_folder='original', # This folder is where the images that are going to be used for the model's training are saved
            feature_to_divide='bubbles', # This folder is where the masks of the specified feature are saved
            divided_images='originalR', # This folder is where the divided images that are going to be used for the model's training are saved (notice that 'divided' means sliced into two subgroups whose diopter's distribution is the same)
            divided_features='bubblesR', # This folder is where the divided images that are going to be used for the model's training are saved.
            doplot=True
        )
    elif f2 == 4:
        automatic_masking.execute(
            dir=dir,
            saved_images_folder='original', # Use either 'original' or 'originalR'
            masks_to_use='bubbles', # Use either 'bubbles' or 'bubblesR'
            using_divided_images=False, # False if 'original' and 'bubbles' specified. True if 'originalR' and 'bubblesR' specified (sclicing performed)
            image_size=128, # Notice that image_size must always be a power of 2. 
            number_of_times_images_will_be_augmented=11,
            name_model_to_save='UNet_muster_model.keras',
            doplot=True
        )
    elif f2 == 5:
        predictions.execute(
            dir=dir,
            saved_images_folder='original', # Use either 'original' or 'originalR'
            using_bubbles_and_differenz=False, # True if the model predicts by extracting bubbles and differenz features (both models must be specified). False if the model predicts by extracting just bubbles features (bubble model must be specified, differenz model can be left as an empty string)
            using_divided_images=False, # False if 'original' and 'bubbles' specified. True if 'originalR' and 'bubblesR' specified (sclicing performed)
            model_bubbles='UNet_Bubbles_size_128_DAS_DE_RED_epoche_50.keras', # Bubble model
            model_differenz='UNet_Segmen_size_128_DAS_DE_RED_epoche_50.keras', # Differenz model
            image_size=128,
            number_of_times_images_will_be_augmented=11,
            model1='model_mlp_1.keras', # Name of the first model to save (notice that the program performs a 5-fold cross validation and therefore trains 5 different models, select the best of them all)
            model2='model_mlp_2.keras',
            model3='model_mlp_3.keras',
            model4='model_mlp_4.keras',
            model5='model_mlp_5.keras',
            doplot=True
        )