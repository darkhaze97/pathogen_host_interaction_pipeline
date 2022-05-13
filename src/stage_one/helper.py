import os

# The function below is to be used with the filter function. It is used in conjunction
# with measure.regionprops, when an intensity image is provided. The intensity image
# must have all labels of 1. This function returns the labels with a mean intensity of 1.
# Arguments:
#   - data (list of RegionProperties)
# returns:
#   - True or False, based on whether the condition: data['intensity_mean'] == 1
#     is met.
def filter_one_hundred_mean_intensity(data):
    if (data['intensity_mean'] == 1):
        return True
    return False

# The function below is similar to the filter_one_hundred_mean_intensity function. This
# function returns the labels with a mean intensity of 0.
# Arguments:
#   - data (list of RegionProperties)
# Returns:
#   - True or False, based on whether the condition: data['intensity_mean'] == 0
#     holds.
def filter_zero_mean_intensity(data):
    if (data['intensity_mean'] == 0):
        return True
    return False
    
# The function is simply a function to check if there exists a path to a user-defined directory.
# If there is not one, then it will print an error.
# arguments:
#   - path (string): The path to the directory
# returns:
#   - None. Simply prints/outputs an error.
def file_exception_checking(path):
    if not (os.path.exists(path)):
        print('The directory path does not exist.')

# The function below takes in a path, and simply obtains the file names in that path.
# This is for reusability - other functions can use this function without having to write
# a try-except block for each of them.
# Arguments:
#   - path (string): The path to a directory
# Returns:
#   - fileNames (list<string>): List of file names
def obtain_file_names(path):
    pathNames = []
    try:
        for fileName in os.listdir(path):
            # Append fileName onto path to form the path to the image
            imagePath = path + f'/{fileName}'
            imagePath = os.path.abspath(imagePath)
            # Convert backwards slash to forward slash
            imagePath = '/'.join(imagePath.split('\\'))
            pathNames.append(imagePath)
    except:
        file_exception_checking(path)
    return pathNames