from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import scipy.io
from scipy import ndimage
from pymatreader import read_mat
from scipy.io import savemat
import math
import copy
import random
from scipy.ndimage import gaussian_filter

def load_data(type, nb_images=3064):
    data =  {}
    if type == 'mat':
        for i in range(nb_images):
            dir = "dataset/Mat_Format_Dataset/" + str(i+1) + ".mat"
            data[str(i+1)] = read_mat(dir)
        return data
    
    if type == 'png':
        class_nb = [1, 2, 3]    
        for i in range(len(class_nb)):
            dir = "dataset/png/" + str(class_nb[i])
            for path in os.listdir(dir):
                data[i] = io.imread(dir + "/" + path)
        return data

def make_couple(data):
    # Verification
    if not( len(data) % 2 == 0):
        print('La dimension des données doit etre paire')
        return
    
    couples = []
    for i in range(int(len(data)/2.0)):
        couples.append([data[2*i], data[2*i + 1]])
    return couples

def flip_horizontal(mat, parameter=None, display=False):
    """
    Flip horizontal d'une seule image (top to bottom)
    input : 
        - mat : donnee au format mat, incluant la matrice rgba, la ROI et le contour
    output : 
        - flipped_mat : copie de la donnee d'entree apres avoir subi une reflexion horizontale
    """
    flipped_mat = copy.deepcopy(mat)
    couleurs = np.array(mat['cjdata']['image'])
    mask = np.array(mat['cjdata']['tumorMask'])
    border = np.array(mat['cjdata']['tumorBorder'])
    border_couple = np.array(make_couple(border))
    # X (HORIZONTAL DIRECTION, COL, ETC) : border_couple[:,0]
    # Y (VERTICAL DIRECTION, ROW, ETC) : border_couple[:,1]
    
    f_couleurs = np.flip(couleurs, axis=0)
    f_mask = np.flip(mask, axis=0)
    flip_axis = couleurs.shape[0] / 2
    f_border_couple = np.array(border_couple)
    f_border_couple[:,1] = 2*flip_axis - f_border_couple[:,1]
    
    flipped_mat = copy.deepcopy(mat)
    flipped_mat['cjdata']['image'] =  f_couleurs
    flipped_mat['cjdata']['tumorMask'] = f_mask
    flipped_mat['cjdata']['tumorBorder'] = f_border_couple.flatten()
    
    if display:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig, axs = plt.subplots(2, 3, figsize=(10,10))
        # original
        axs[0,0].imshow(couleurs)
        axs[0,0].title.set_text('Couleurs orginales')
        axs[0,1].imshow(mask, cmap='Greys')
        axs[0,1].title.set_text('Mask orginal')
        axs[0,2].scatter(border_couple[:,0], border_couple[:,1], s=20)
        axs[0,2].plot(border_couple[:,0], border_couple[:,1])
        axs[0,2].set_xlim([0,512])
        axs[0,2].set_ylim([0,512])
        axs[0,2].invert_yaxis()
        axs[0,2].set(adjustable='box', aspect='equal')
        axs[0,2].title.set_text('Contour original')
        # flipped
        axs[1,0].imshow(f_couleurs)
        axs[1,0].title.set_text('Couleurs flipped')
        axs[1,1].imshow(f_mask, cmap='Greys')
        axs[1,1].title.set_text('Mask flipped')
        axs[1,2].scatter(f_border_couple[:,0], f_border_couple[:,1],s=20)
        axs[1,2].plot(f_border_couple[:,0], f_border_couple[:,1])
        axs[1,2].set_xlim([0,512])
        axs[1,2].set_ylim([0,512])
        axs[1,2].invert_yaxis()
        axs[1,2].set(adjustable='box', aspect='equal')
        axs[1,2].title.set_text('Contour flipped')
    
    return flipped_mat

def flip_vertical(mat, parameter=None, display=False):
    """
    Flip vertical d'une seule image (top to bottom)
    input : 
        - mat : donnee au format mat, incluant la matrice rgba, la ROI et le contour
    output : 
        - flipped_mat : copie de la donnee d'entree apres avoir subi une reflexion verticale
    """
    flipped_mat = copy.deepcopy(mat)
    couleurs = np.array(mat['cjdata']['image'])
    mask = np.array(mat['cjdata']['tumorMask'])
    border = np.array(mat['cjdata']['tumorBorder'])
    border_couple = np.array(make_couple(border))
    # X (HORIZONTAL DIRECTION, COL, ETC) : border_couple[:,0]
    # Y (VERTICAL DIRECTION, ROW, ETC) : border_couple[:,1]
    
    f_couleurs = np.flip(couleurs, axis=1)
    f_mask = np.flip(mask, axis=1)
    flip_axis = couleurs.shape[1] / 2
    f_border_couple = np.array(border_couple)
    f_border_couple[:,0] = 2*flip_axis - f_border_couple[:,0]
    
    flipped_mat['cjdata']['image'] =  f_couleurs
    flipped_mat['cjdata']['tumorMask'] = f_mask
    flipped_mat['cjdata']['tumorBorder'] = f_border_couple.flatten()
    
    if display:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig, axs = plt.subplots(2, 3, figsize=(10,10))
        axs[0,0].imshow(couleurs)
        axs[0,0].title.set_text('Couleurs orginales')
        axs[0,1].imshow(mask, cmap='Greys')
        axs[0,1].title.set_text('Mask orginal')
        axs[0,2].scatter(border_couple[:,0], border_couple[:,1], s=20)
        axs[0,2].plot(border_couple[:,0], border_couple[:,1], )
        axs[0,2].set_xlim([0,512])
        axs[0,2].set_ylim([0,512])
        axs[0,2].invert_yaxis()
        axs[0,2].set(adjustable='box', aspect='equal')
        axs[0,2].title.set_text('Contour original')

        axs[1,0].imshow(f_couleurs)
        axs[1,0].title.set_text('Couleurs flipped')
        axs[1,1].imshow(f_mask, cmap='Greys')
        axs[1,1].title.set_text('Mask flipped')
        axs[1,2].scatter(f_border_couple[:,0], f_border_couple[:,1],s=20)
        axs[1,2].plot(f_border_couple[:,0], f_border_couple[:,1])
        axs[1,2].set_xlim([0,512])
        axs[1,2].set_ylim([0,512])
        axs[1,2].invert_yaxis()
        axs[1,2].set(adjustable='box', aspect='equal')
        axs[1,2].title.set_text('Contour flipped')
    
    return flipped_mat

def rotate_2d_points(p, origin=(0, 0), degrees=0):
    """
    angle is in radian
    """
    # source : https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python#:~:text=def%20rotate(p,10)%0Aprint(new_points)
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotate_90(mat, n=1, display=False):
    """
    Rotation d'une seule image par intervalle de 90 degres
    input : 
        - mat : donnee au format mat, incluant la matrice rgba, la ROI et le contour
        - n : nombre de rotation de 90 degres qui sera effectuee sur l'imag
    output : 
        - rotated_mat : copie de la donnee d'entree apres avoir subi une rotation
    """
    
    r_mat = copy.deepcopy(mat)
    couleurs = np.array(mat['cjdata']['image'])
    mask = np.array(mat['cjdata']['tumorMask'])
    border = np.array(mat['cjdata']['tumorBorder'])
    border_couple = np.array(make_couple(border))
    # X (HORIZONTAL DIRECTION, COL, ETC) : border_couple[:,0]
    # Y (VERTICAL DIRECTION, ROW, ETC) : border_couple[:,1]
    
    r_couleurs = np.rot90(couleurs, n)
    r_mask = np.rot90(mask, n)
    r_mask = np.rot90(mask, n)
    rotation_origin = (couleurs.shape[0] / 2, couleurs.shape[1] / 2) 
    r_border_couple = rotate_2d_points(border_couple, rotation_origin, -90*n)
    
    r_mat['cjdata']['image'] =  r_couleurs
    r_mat['cjdata']['tumorMask'] = r_mask
    r_mat['cjdata']['tumorBorder'] = r_border_couple.flatten()
    
    if display:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig, axs = plt.subplots(2, 3, figsize=(10,10))
        axs[0,0].imshow(couleurs)
        axs[0,0].title.set_text('Couleurs orginales')
        axs[0,1].imshow(mask, cmap='Greys')
        axs[0,1].title.set_text('Mask orginal')
        axs[0,2].scatter(border_couple[:,0], border_couple[:,1], s=20)
        axs[0,2].plot(border_couple[:,0], border_couple[:,1], )
        axs[0,2].set_xlim([0,512])
        axs[0,2].set_ylim([0,512])
        axs[0,2].invert_yaxis()
        axs[0,2].set(adjustable='box', aspect='equal')
        axs[0,2].title.set_text('Contour original')

        axs[1,0].imshow(r_couleurs)
        axs[1,0].title.set_text('Couleurs rotated '+ str(90*n) + ' degres')
        axs[1,1].imshow(r_mask, cmap='Greys')
        axs[1,1].title.set_text('Mask rotated ' + str(90*n) + ' degres')
        axs[1,2].scatter(r_border_couple[:,0], r_border_couple[:,1],s=20)
        axs[1,2].plot(r_border_couple[:,0], r_border_couple[:,1])
        axs[1,2].set_xlim([0,512])
        axs[1,2].set_ylim([0,512])
        axs[1,2].invert_yaxis()
        axs[1,2].set(adjustable='box', aspect='equal')
        axs[1,2].title.set_text('Contour rotated ' + str(90*n) + ' degres')
    
    return r_mat

def gaussian_blur(mat, sigma=[2, 0, 0], display=False):
    """
    Application d'un filtre gaussien sur une seule image 
    input : 
        - mat : donnee au format mat, incluant la matrice rgba, la ROI et le contour
    output : 
        - gaussian_mat : copie de la donnee d'entree sur laquelle on a applique un flou gaussien
    """
    
    blurred_mat = copy.deepcopy(mat)
    couleurs = np.array(mat['cjdata']['image'])
    mask = np.array(mat['cjdata']['tumorMask'])
    border = np.array(mat['cjdata']['tumorBorder'])
    border_couple = np.array(make_couple(border))
    # X (HORIZONTAL DIRECTION, COL, ETC) : border_couple[:,0]
    # Y (VERTICAL DIRECTION, ROW, ETC) : border_couple[:,1]
    
    b_couleurs = gaussian_filter(couleurs, sigma=sigma[0])
    b_mask = gaussian_filter(mask, sigma=sigma[1])
    b_border_couple = gaussian_filter(border_couple, sigma=sigma[2])
    
    blurred_mat['cjdata']['image'] =  b_couleurs
    blurred_mat['cjdata']['tumorMask'] = b_mask
    blurred_mat['cjdata']['tumorBorder'] = b_border_couple.flatten()
    
    if display:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig, axs = plt.subplots(2, 3, figsize=(10,10))
        axs[0,0].imshow(couleurs)
        axs[0,0].title.set_text('Couleurs orginales')
        axs[0,1].imshow(mask, cmap='Greys')
        axs[0,1].title.set_text('Mask orginal')
        axs[0,2].scatter(border_couple[:,0], border_couple[:,1], s=20)
        axs[0,2].plot(border_couple[:,0], border_couple[:,1], )
        axs[0,2].set_xlim([0,512])
        axs[0,2].set_ylim([0,512])
        axs[0,2].invert_yaxis()
        axs[0,2].set(adjustable='box', aspect='equal')
        axs[0,2].title.set_text('Contour original')

        axs[1,0].imshow(b_couleurs)
        axs[1,0].title.set_text('Couleurs blurred')
        axs[1,1].imshow(b_mask, cmap='Greys')
        axs[1,1].title.set_text('Mask blurred')
        axs[1,2].scatter(b_border_couple[:,0], b_border_couple[:,1],s=20)
        axs[1,2].plot(b_border_couple[:,0], b_border_couple[:,1])
        axs[1,2].set_xlim([0,512])
        axs[1,2].set_ylim([0,512])
        axs[1,2].invert_yaxis()
        axs[1,2].set(adjustable='box', aspect='equal')
        axs[1,2].title.set_text('Contour blurred')
    
    return blurred_mat

def classify_data(mat_data):
    """
    input:
        - mat_data : dictionnaire de donnees a separer {}
    ouput:
        - classified_lists: : listes de dictionnaire separes selon le type de tumeur [ {}, {}, {} ]
    """
    meningioma = {}
    glioma = {}
    pituitary = {}
    classified_lists = [meningioma, glioma, pituitary]

    for key, value in mat_data.items():
        class_index = int(value['cjdata']['label'] - 1)
        classified_lists[class_index][key] = value
    
    return classified_lists

def random_sampling(mat_data, n):
    """
    input:
        - mat_data : dictionnaire de donnees a echantillonner {}
        - n : nombre de donnees a echantillonner
    output:
        - dictionnaire contenant les donnees echantillonnees
    """
    # On remplit le dictionnaire 
    sampling = {}
    while len(sampling) < n:
        random_key = random.choice(list(mat_data))
        # On ajoute la clee aleatoire au dictionnaire de sampling
        sampling[random_key] = copy.deepcopy(mat_data[random_key])
        # On retire la clee aleatoire du dictionnaire original
        del mat_data[random_key]
    return sampling

def separate_train_test_val(mat_data, p_train=0.8, p_test=0.1, p_val=0.1, save=False):
    """
    input:
        - mat_data : liste de donnees a separer [ {}, {}, {} ]
        - p_train : pourcentage de donnees en train
        - p_test : pourcentage de donnees en test
        - p_val : pourcentage de donnees en validation
    output:
        - mat_train : donnees d'entrainement
        - mat_test : donnees de test
        - mat_val : donnees de validation
    """
    n_class = 3
    N = len(mat_data[0]) + len(mat_data[1]) + len(mat_data[2])
    N_train = int(p_train * N)
    N_test = int(p_test * N)
    N_val = int(p_val * N)
    
    mat_test = [{}, {}, {}]
    mat_val = [{}, {}, {}]
    for i in range(n_class):
        mat_test[i] = random_sampling(mat_data[i], int(N_test/n_class))
        mat_val[i] = random_sampling(mat_data[i], int(N_val/n_class))
    # remainder goes to train
    mat_train = mat_data
    # for i in range(n_class):
    #     while len(mat_data[i]) > 0:
    #         mat_train.append(mat_data[i].pop(0))
    
    # sanity check
    # if  len(mat_train) < (N_train - 2*n_class)  \
    #     or len(mat_test) < (N_test - 2*n_class) \
    #     or len(mat_val) < (N_val - 2*n_class) :
    #     print('Something went wrong')
    #     print('N', N)
    #     print('N_test', N_test)
    #     print('len(mat_test)', len(mat_test))
    #     print('N_val', N_val)
    #     print('len(mat_val)', len(mat_val))
    #     print('N_train', N_train)
    #     print('len(mat_train)', len(mat_train))
        # return
    # sanity check
    if abs(len(mat_test[0]) + len(mat_test[1]) + len(mat_test[2]) - N_test) > 1 :
        print('Something went wrong')
        print('N_test', N_test)
        print('len(mat_test)', len(mat_test[0]) + len(mat_test[1]) + len(mat_test[2]))
        return
    
    if abs(len(mat_val[0]) + len(mat_val[1]) + len(mat_val[2]) - N_val) > 1 :
        print('Something went wrong')
        print('N_val', N_val)
        print('len(mat_val)', len(mat_val[0]) + len(mat_val[1]) + len(mat_val[2]))
        return

    if  abs(len(mat_train[0]) + len(mat_train[1]) + len(mat_train[2]) - N_train) > 1 :
        print('Something went wrong')
        print('N_train', N_train)
        print('len(mat_train)', len(mat_train[0]) + len(mat_train[1]) + len(mat_train[2]))
        return
    
    if save:
        # Train must me equally distributed among classes before saving
        # save_data(mat_train, 'dataset/Train')
        save_data(mat_test, 'dataset/Test')
        save_data(mat_val, 'dataset/Validation')
    
    return mat_train, mat_test, mat_val

def save_data(mat_data, dir_name):
    """
    input:
        - mat_data : liste de donnees a sauvegarder {}
        - dir_name : dossier ou les donnees seront sauvegardees
    output:
        - aucun, mais les donnees sont sauvegardes dans les dossiers respectifs (Test, Train, Validation)
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not len(os.listdir(dir_name)) == 0:
        print("WARNING: Directory is not empty")
        # return

    n_data = len(mat_data)
    count = 0
    for index, value in mat_data.items():
        if count == int(n_data/4):
            print('25% complete')
        if count == int(n_data/2) < 1:
            print('50% complete')
        if count == 3*n_data/4:
            print('75% complete')
        if count == (n_data - 1):
            print('100% complete')
        
        file_name = dir_name + '/' + str(index) + '.mat'
        savemat(file_name, value)

def generate_flip_horizontal(mat_data):
    print('Flip horizontal de donnees')
    print('0% complete')
    transformation = "flip_horizontal"
    for key, value in mat_data.items():        
        if key == 1000:
            print('33% complete')
        if key == 2000:
            print('66% complete')

        fileName = "dataset/" + transformation +  "/"  + str(key) + ".mat"
        savemat(fileName, flip_horizontal(value))
    print('100% complete')

def generate_flip_vertical(mat_data):
    print('Flip vertical de donnees')
    print('0% complete')
    transformation = "flip_vertical"
    for key, value in mat_data.items():        
        if key == 1000:
            print('33% complete')
        if key == 2000:
            print('66% complete')

        fileName = "dataset/" + transformation + "/"  + str(key) + ".mat"
        savemat(fileName, flip_vertical(value))
    print('100% complete')

# Cette fonction genere les donnees "horizontally flipped" pour tous les elements de mat_data
def generate_rot(mat_data, n):

    if n == 1:
        transformation = "rot_90"
    if n == 2:
        transformation = "rot_180"
    if n == 3:
        transformation = "rot_270"
    print('Rotation de donnees (' + transformation + ')')
    print('0% complete')
    for key, value in mat_data.items():        
        if key == 1000:
            print('33% complete')
        if key == 2000:
            print('66% complete')

        fileName = "dataset/" + transformation + "/"  + str(key) + ".mat"
        savemat(fileName, rotate_90(value, n))
    print('100% complete')

def generate_gaussian(mat_data):
    print('Bruit gaussien')
    print('0% complete')
    transformation = 'gaussian'
    for key, value in mat_data.items():        
        if key == 1000:
            print('33% complete')
        if key == 2000:
            print('66% complete')

        fileName = "dataset/" + transformation + "/"  + str(key) + ".mat"
        savemat(fileName, gaussian_blur(value))
    print('100% complete')

class DataSeparator:

    def __init__(self, mat_data, equalizing_transformations , p_train=0.8, p_test=0.1, p_val=0.1):
        """
        - mat_data : liste de donnees a separer [{}, {}, {}], divisees en classes
        - equalizing_transformations : liste des combinaisons de transformations a appliquer [Transformation1, Transformation2, ... , TransformationN]
        - p_train : pourcentage de donnees a separer en train
        - p_test : pourcentage de donnees a separer en test
        - p_val : pourcentage de donnees a separer en validation
        """
        self.mat_data = mat_data
        self.mat_val = [{}, {}, {}]
        self.mat_test = [{}, {}, {}]
        self.mat_train = [{}, {}, {}]
        self.equalizing_transformations = equalizing_transformations
        self.N = len(mat_data[0]) + len(mat_data[1]) + len(mat_data[2])
        self.p_train = p_train
        self.p_test = p_test
        self.p_val = p_val
        self.n_train = int(self.N * p_train)
        self.n_test = int(self.N * p_test)
        self.n_val = int(self.N * p_val)

    def separate_data(self, save=False):
        """
        - separe les donnees en train, test et validation
        - les donnees seront separees en 3 ensembles :
            - train
            - test
            - validation
        """
        # Separation des donnees en train, test et validation
        self.mat_train, self.mat_test, self.mat_val = separate_train_test_val(self.mat_data, self.p_train, self.p_test, self.p_val, save)
    
    def equalize_train(self):
        """
        Augmente les donnees d'entrainement jusqu'a ce que chaque classe soit egalement representee dans le train
        """
        # nombre de donnees supplementaires requises pour chaque classe
        n_par_classe = [len(self.mat_train[0]), len(self.mat_train[1]), len(self.mat_train[2])]
        print('n_par_classe: ', n_par_classe)
        n_max = max(n_par_classe)
        n_requis = [n_max - n_par_classe[0], n_max - n_par_classe[1], n_max - n_par_classe[2]]
        print('n_requis: ', n_requis)

        # Validation pre egalisation
        if n_par_classe[0] * len(self.equalizing_transformations) < n_max:
            print('Impossible d\'augmenter les donnees d\'entrainement')
            print('n_classe_1 *  len(self.equalizing_transformations) = ' + str(n_par_classe[0] * len(self.equalizing_transformations)))
            print('n_max = ' + str(n_max))

        if n_par_classe[1] * len(self.equalizing_transformations) < n_max:
            print('Impossible d\'augmenter les donnees d\'entrainement')
            print('n_classe_1 *  len(self.equalizing_transformations) = ' + str(n_par_classe[1] * len(self.equalizing_transformations)))
            print('n_max = ' + str(n_max))
        
        if n_par_classe[2] * len(self.equalizing_transformations) < n_max:
            print('Impossible d\'augmenter les donnees d\'entrainement')
            print('n_classe_2 *  len(self.equalizing_transformations) = ' + str(n_par_classe[2] * len(self.equalizing_transformations)))
            print('n_max = ' + str(n_max))

        # Augmentation des donnees d'entrainement
        # iteration sur chaque classe
        for c in range(len(self.mat_train)):
            if n_requis[c] > 0:
                print('Augmentation des donnees d\'entrainement de la classe ' + str(c))
                keys = list(self.mat_train[c])
                for i in range(n_requis[c]):
                    if i == int(n_requis[c] / 4):
                        print('25% complete')
                    if i == int(n_requis[c] / 2):
                        print('50% complete')
                    if i == int(n_requis[c] * 3 / 4):
                        print('75% complete')
                    if i == (n_requis[c] - 1):
                        print('100% complete')
                    key = keys[i % len(keys)]
                    value = self.mat_train[c][key]
                    transformation = self.equalizing_transformations[int(i / n_par_classe[c])]
                    self.mat_train[c][key + '_eq_' + transformation.get_code()] = transformation.apply_one_transformation(value)
            else:
                print('Les donnees de la classe ' + str(c) + ' sont suffisantes')

    def augment_train(self, transformation):
        """
        Augmente les donnees d'entrainement
        - transformation : transformation a appliquer (type : Transformation)
        """
        for c in range(len(self.mat_train)):
            print('Augmentation des donnees d\'entrainement de la classe ' + str(c))
            print('Transformation: ' + transformation.get_code())
            self.mat_train[c].update(transformation.apply_transformation(self.mat_train[c]))

    def save_to_file_system(self):
        """
        Sauvegarde les donnees dans le systeme de fichiers
        """
        for c in range(len(self.mat_train)):
            print('Sauvegarde des donnees de validation de la classe ' + str(c))
            save_data(self.mat_val[c], 'dataset/Validation')
            print('Sauvegarde des donnees de test de la classe ' + str(c))
            save_data(self.mat_test[c], 'dataset/Test')
            print('Sauvegarde des donnees d\'entrainement de la classe ' + str(c))
            save_data(self.mat_train[c], 'dataset/train')

class Transformation:
    def __init__(self, transformation_function, parameters=None, display=False):
        """
        - transformation_function : fonction de transformation
        - parameters : liste des parametres de la transformation
        - display : boolean, True si on veut afficher les transformations
        """
        self.transformation_function = transformation_function
        self.parameters = parameters
        self.display = display
    
    def apply_transformation(self, mat_data):
        """
        Applique la transformation sur les donnees
        - mat_data : dictionnaire de donnees a transformer
        """
        transformed_data = {}
        count = 0
        for key, value in mat_data.items():        
            if count == int(len(mat_data) / 4):
                print('25% complete')
            if count == int(len(mat_data) / 2):
                print('50% complete')
            if count == int(len(mat_data) * 3 / 4):
                print('75% complete')
            if count == (len(mat_data) - 1):
                print('100% complete')

            if self.parameters == None:
                transformed_data[key + '_aug_' + self.get_code()] = self.transformation_function(value)
            else:
                transformed_data[key + '_aug_' + self.get_code()] = self.transformation_function(value, self.parameters)
            count = count + 1
        
        return transformed_data

    def apply_one_transformation(self, mat_data):
        """
        Applique la transformation sur une seule donnee
        """
        if self.parameters == None:
            return self.transformation_function(mat_data)
        else:
            return self.transformation_function(mat_data, self.parameters)

    def get_code(self):
        """
        Retourne le code de la transformation
        """
        if self.parameters == None:
            return self.transformation_function.__name__[0:6]
        else:
            return self.transformation_function.__name__[0:6] + '_' + str(self.parameters)
    
    