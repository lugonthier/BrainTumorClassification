import numpy as np
import tensorflow as tf
from data_loader.loader import DataLoader
from models.unet import UNet
from models.vgg import VGG
from models.classifier import Classifier
from sklearn.model_selection import train_test_split



if __name__=="__main__":
    # Loading data
    print("Loading data\n\n")
    dl = DataLoader()
    dataset = dl.load_data("/content/BrainTumorDataset/Mat_Format_Dataset/", "mat", height=160, width=160)
    print("Convert to numpy\n\n")
    dataset = dl.configure_for_performance(dataset, shuffle=True)
    
    