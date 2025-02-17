import os
import glob

BATCH_SIZE = 32
IMG_HEIGHT = 224 # after padding (we'll pad )
IMG_WIDTH = 224  # after padding
IMG_CHANNELS = 3 # we will increase if need be to match this


PROB_THRESHOLD = 0.5 # probabilities above this will be considered tortuous


# list of images, that we we want to test against the model
# we will traverse the original image file folder in original_images

original_img_files = glob.glob(os.path.join("model_eval/original_images", "*"))
# we will assume that there is a folder named same as the original file path containing the datasets as
# -images
#       -{image file name as folder name}
#           -dataset
#               -tortuous
#               -non_tortuous

# csv path log files will exist in the csv folder
#   -csv
#       -{image_filename}.csv

if not os.path.exists("model_eval/images"): 
    os.makedirs("model_eval/images") 

data_dirs = [] 
for i, paths in enumerate(original_img_files):
    # we are using 'result' for now, but later we will change it to 'dataset'
    data_dirs.append(os.path.join("model_eval/images", original_img_files[i].split("\\")[1].split('.')[0], "result"))

if not os.path.exists("model_eval/result"): 
    os.makedirs("model_eval/result") 

if not os.path.exists("model_eval/final_csvs"): 
    os.makedirs("model_eval/final_csvs") 