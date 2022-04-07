from DataTransformation import *
import importlib


if __name__=="__main__":
    # PARAMS
    augment_train = True


    mat_data = load_data('mat')
    classified_data = classify_data(mat_data)
    print(len(classified_data[0])) # 708
    print(len(classified_data[1])) # 1426
    print(len(classified_data[2])) # 930
    total = len(classified_data[0]) + len(classified_data[1]) + len(classified_data[2])
    print(total)
    equalizing_transformations = [Transformation(rotate_90), Transformation(flip_horizontal), Transformation(flip_vertical)]
    augmentation_transformations = [Transformation(rotate_90)]
        
    dataSeparator = DataSeparator(classified_data, equalizing_transformations)

    # Splitting
    dataSeparator.separate_data(save=False)
    # Save original train.
    dataSeparator.save_train_to_file_system('train_original')

    dataSeparator.equalize_train()

    dataSeparator.save_train_to_file_system('train_equalize')

    # Augmentation
    if augment_train:
        dataSeparator.augment_train(Transformation(gaussian_blur))
        #dataSeparator.augment_train(Transformation(rotate_90))

    dataSeparator.save_to_file_system()