import pandas as pd
import numpy as np
import cv2
import csv
from progress.bar import Bar


def load_dataframe(path_csv):
    """
    Load a single dataframe from an csv file
    """
    return pd.read_csv(path_csv)


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


def randomize_data(dataframe, seed):
    """
    Returns a randomized version of the input data based on the seed
    """
    np.random.seed(seed)
    print("Randomizing..")
    copy = dataframe.copy()
    np.random.shuffle(copy.values)
    return copy


if __name__ == '__main__':
    # load data
    X_train = load_dataframe('../data/train/X_train_gr_smpl.csv')
    y_train = load_dataframe('../data/train/y_train_smpl.csv')
    X_test = load_dataframe('../data/test/X_test_gr_smpl.csv')
    y_test = load_dataframe('../data/test/y_test_smpl.csv')
    # My 3, Mase Pos and Me
    random_seed = 3
    # randomize
    X_train = randomize_data(X_train, random_seed)
    y_train = randomize_data(y_train, random_seed)
    X_test = randomize_data(X_test, random_seed)
    y_test = randomize_data(y_test, random_seed)
    # reduce size by slicing image
    X_train_slc = slice_img(X_train)
    X_test_slc = slice_img(X_test)
    # normalize image
    X_train_slc_nrml = normalize(X_train_slc)
    X_test_slc_nrml = normalize(X_test_slc)
    # save your work!
    save_dataframe_csv(X_train_slc_nrml, '../data/train/X_train_sliced_n_diced.csv')
    save_dataframe_csv(y_train, '../data/train/y_train_rnd.csv')
    save_dataframe_csv(X_test_slc_nrml, '../data/test/X_test_sliced_n_diced.csv')
    save_dataframe_csv(y_test, '../data/test/y_test_rnd.csv')
    # take 4000 instances of the training set and give them to the test set
    X_train_slc_nrml_less_4000 = X_train_slc_nrml[:-4000]
    y_train_less_4000 = y_train[:-4000]
    X_test_slc_nrml_up_4000 = pd.concat([X_test_slc_nrml, X_train_slc_nrml[-4000:]])
    y_test_up_4000 = pd.concat([y_test, y_train[-4000:]])
    save_dataframe_csv(X_train_slc_nrml_less_4000, '../data/train/X_train_sliced_n_diced_less_4000.csv')
    save_dataframe_csv(y_train_less_4000, '../data/train/y_train_rnd_less_4000.csv')
    save_dataframe_csv(X_test_slc_nrml_up_4000, '../data/test/X_test_sliced_n_diced_up_4000.csv')
    save_dataframe_csv(y_test_up_4000, '../data/test/y_test_rnd_up_4000.csv')
    # at this rate there will be no training set left!
    X_train_slc_nrml_less_9000 = X_train_slc_nrml[:-9000]
    y_train_less_9000 = y_train[:-9000]
    X_test_slc_nrml_up_9000 = pd.concat([X_test_slc_nrml, X_train_slc_nrml[-9000:]])
    y_test_up_9000 = pd.concat([y_test, y_train[-9000:]])
    save_dataframe_csv(X_train_slc_nrml_less_9000, '../data/train/X_train_sliced_n_diced_less_9000.csv')
    save_dataframe_csv(y_train_less_9000, '../data/train/y_train_rnd_less_9000.csv')
    save_dataframe_csv(X_test_slc_nrml_up_9000, '../data/test/X_test_sliced_n_diced_up_9000.csv')
    save_dataframe_csv(y_test_up_9000, '../data/test/y_test_rnd_up_9000.csv')
