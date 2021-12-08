from utils import device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from icecream import ic

# # Nombre de stations utilisé
# CLASSES = 10
# #Longueur des séquences
# LENGTH = 20
# # Dimension de l'entrée (1 (in) ou 2 (in/out))
# DIM_INPUT = 2
# #Taille du batch
# BATCH_SIZE = 32

# PATH = "/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/AMAL-student_tp4.2021/data/"

# matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
# train_data = (SampleMetroDataset(matrix_train[:,:,:CLASSES,:DIM_INPUT],length=LENGTH))
# train_dataloader =DataLoader((train_data),batch_size=BATCH_SIZE,shuffle=False)
# data_test = DataLoader(SampleMetroDataset(matrix_test[:,:,:CLASSES,:DIM_INPUT],length=LENGTH,stations_max=train_data.stations_max), batch_size=BATCH_SIZE,shuffle=False)
# train_features, train_labels = next(iter(train_dataloader))
# ic(train_features.shape)
# ic(train_labels.shape)
# #  TODO:  Question 2 : prédiction de la ville correspondant à une séquence


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6