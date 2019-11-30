import csv
import cv2
import numpy as np
import pandas as pd
import data_ops
from progress.bar import Bar

# Use opnenc cv to read images, resize them and apply grey filter
# After that, we want to save those pixels as csv files.


def load_image(path):
    print("Loading images...")
    img = cv2.imread(path)
    print("Converting to gray...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Done")
    return gray
    
def convert_image(image, path):
    data = np.asarray(image, dtype=np.int).flatten()
    print("Passing image pixels to csv...")
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        print("File created")
    return data

def slice_imgae(image):
    bar = Bar('Slicing', max=len(dataframe.values))
    data = []
    re = np.reshape(image, (48, 48))
    sub_matrix = re[9:37, 9:37]
    data.append(sub_matrix.flatten())
    bar.next()
    reduced = pd.DataFrame(data, columns=range(0, 28**2))
    bar.finish()
    return reduced

# Operations from data_operations for reducing, slicing and randomizing the images

def normalize(dataframe):
    """
    Create the normalized version of the train_smpl
    The image's pixels are converted in a range [0-255]
    """
    normalized = []
    bar = Bar('Normalizing\t', max=len(dataframe.values))
    for pixels in dataframe.values:
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pixels)
        new_row = []
        for pixel in pixels:
            pixel = int((pixel - minVal) * (255 / (maxVal - minVal)))
            new_row.append(pixel)
        normalized.append(new_row)
        bar.next()
    bar.finish()
    return pd.DataFrame(normalized, columns=dataframe.columns)


if __name__ == '__main__':
    #load data
    new_test_1 = load_image("../scripts/data/new_test/test1_px.png")
    new_test_2 = load_image("../scripts/data/new_test/test2_px.png")
    new_test_2 = load_image("../scripts/data/new_test/test3_px.png")
    
    #save my work
    test1_csv = convert_image(new_test_1, "../scripts/data/new_test/test1_px.csv")
    test2_csv = convert_image(new_test_1, "../scripts/data/new_test/test2_px.csv")
    test3_csv = convert_image(new_test_1, "../scripts/data/new_test/test3_px.csv")

    #Load dataframe
    x_test1 = data_ops.load_dataframe("../scripts/data/new_test/test1_px.csv")
    x_test2 = data_ops.load_dataframe("../scripts/data/new_test/test2_px.csv")
    x_test3 = data_ops.load_dataframe("../scripts/data/new_test/test3_px.csv")
    
    #Randomise pixels
    random_seed = 3
    x_test1 = data_ops.randomize_data(x_test1, random_seed)
    x_test2 = data_ops.randomize_data(x_test2, random_seed)
    x_test3 = data_ops.randomize_data(x_test3, random_seed)
    
    #Reduce size of image by slicing it
    x_test1_slc = data_ops.slice_img(x_test1)
    x_test2_slc = data_ops.slice_img(x_test2)
    x_test3_slc = data_ops.slice_img(x_test3)
    
    #Normalize image
    x_test1_slc_nrml = normalize(x_test1_slc)
    x_test2_slc_nrml = normalize(x_test1_slc)
    x_test3_slc_nrml = normalize(x_test1_slc)
    
    #Save the images in new csv files
    data_ops.save_dataframe_csv(x_test1_slc_nrml, "../scripts/data/new_test/x_test1_sliced_n_diced.csv")
    data_ops.save_dataframe_csv(x_test1, "../scripts/data/new_test/x_test1_rnd.csv")

    data_ops.save_dataframe_csv(x_test2_slc_nrml, "../scripts/data/new_test/x_test2_sliced_n_diced.csv")
    data_ops.save_dataframe_csv(x_test2, "../scripts/data/new_test/x_test2_rnd.csv")

    data_ops.save_dataframe_csv(x_test3_slc_nrml, "../scripts/data/new_test/x_test3_sliced_n_diced.csv")
    data_ops.save_dataframe_csv(x_test3, "../scripts/data/new_test/x_test3_rnd.csv")

# Operations 
# Now onto the tree operations and NN.


"""
if__
def convert_to_csv()
data = []
data.append(gray.flatten())
print(data)

# Save Greyscale values
data = np.asarray(data, dtype=np.int)
data = data.flatten()
print(data)
with open("../scripts/data/new_test/test1.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(data)


#cv2.imshow('Original image',img)
#cv2.imshow('Gray image', gray)
#cv2.imwrite("../scripts/data/new_test/test1_gray.png", gray)
#cv2.waitKey(0)

"""
