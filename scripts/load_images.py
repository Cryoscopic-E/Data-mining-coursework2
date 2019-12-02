import csv
import cv2
import numpy as np
import pandas as pd
import data_ops
from progress.bar import Bar


# Use opnencv to load image and apply grey scale filter. After that, we convert it to a .csv file

def load_image(path, path_grey):
    img = cv2.imread(path)  # Load image
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) # Convert to gray
    cv2.imwrite(path_grey, gray) # Save image in grey scale
    return gray
    
def convert_image(* args):
    _set=list()
    for image in args:
        data = np.asarray(image, dtype=np.int).flatten()
        _set.append(data)
    with open('data/new_test/tests.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(48*48))
        writer.writerows(_set)
        print("File created")
    return data

# Operations from data_operations for reducing, slicing and randomizing the images

def normalize(dataframe):
    """
    Create the normalized version of the test sample
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


def slice_img(dataframe):
    """
    Return the dataframe with all the images reduced to 28x28 to eliminate the background
    """
    bar = Bar('Slicing', max=len(dataframe.values))
    data = []
    for image in dataframe.values:
        re = np.reshape(image, (48, 48))
        sub_matrix = re[9:37, 9:37]
        data.append(sub_matrix.flatten())
        bar.next()
    reduced = pd.DataFrame(data, columns=range(0, 28 ** 2))
    bar.finish()
    return reduced

def save_dataframe_csv(dataframe, file_path):
    """
    Helper method to save a dataframe in the correct format
    """
    if file_path != "":
        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(dataframe.columns)
            bar = Bar('Saving csv\t', max=len(dataframe.values))
            for el in dataframe.values:
                csv_writer.writerow(el)
                bar.next()
            bar.finish()


if __name__ == '__main__':
    #load data
    new_test_1 = load_image('data/new_test/test_1.png', 'data/new_test/test1_grey.png')
    new_test_2 = load_image('data/new_test/test_2.png', 'data/new_test/test2_grey.png')
    new_test_3 = load_image('data/new_test/test_3.png', 'data/new_test/test3_grey.png')
    
    #save my work and combine
    tests_csv = convert_image(new_test_1, new_test_2, new_test_3)


    # combine image

    
    #Load dataframe
    x_test = pd.read_csv('data/new_test/tests.csv')

    #Reduce size of image by slicing it
    x_test_slc = slice_img(x_test)
    
    

    # Normalize pixels
    x_test_normalized = normalize(x_test_slc)

    
    #Save the images in new csv files
    save_dataframe_csv(x_test, 'data/new_test/x_test_t.csv')
    save_dataframe_csv(x_test_slc, 'data/new_test/x_test_sliced.csv')
    save_dataframe_csv(x_test_normalized, 'data/new_test/x_test_normalized.csv')
