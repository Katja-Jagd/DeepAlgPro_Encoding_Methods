#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn
from torch import nn
import os
import numpy as np
import json
from Bio import SeqIO 

class SequenceEncoding:
    encoding_types = ['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                      'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                      'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec']
    residue_types = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
            
    def __init__(self, encoding_type="One_hot"):
        if encoding_type not in SequenceEncoding.encoding_types:
            raise Exception("Encoding type \'%s\' not found" % encoding_type)
        self.encoding_type = encoding_type
        
    def get_ProtVec_encoding(self, ProtVec, seq, overlap=True):
        if overlap:
            encodings = []
            for i in range(len(seq)-2):
                encodings.append(ProtVec[''.join(seq[i:i+3])]) if ProtVec.__contains__(''.join(seq[i:i+3])) else encodings.append(ProtVec["<unk>"])
        else:
            encodings_1, encodings_2, encodings_3 = [], [], []
            for i in range(0, len(seq), 3):
                if i+3 <= len(seq):
                    encodings_1.append(ProtVec[''.join(seq[i:i+3])]) if ProtVec.__contains__(''.join(seq[i:i+3])) else encodings_1.append(ProtVec["<unk>"])
                if i+4 <= len(seq):
                    encodings_2.append(ProtVec[''.join(seq[i+1:i+4])]) if ProtVec.__contains__(''.join(seq[i+1:i+4])) else encodings_2.append(ProtVec["<unk>"])
                if i+5 <= len(seq):
                    encodings_3.append(ProtVec[''.join(seq[i+2:i+5])]) if ProtVec.__contains__(''.join(seq[i+2:i+5])) else encodings_3.append(ProtVec["<unk>"])

            encodings = [encodings_1, encodings_2, encodings_3]
        return encodings

    def get_encoding(self, seq, data_directory, overlap=True):
        #seq = seq.upper()
        with open(data_directory + "%s.json" % self.encoding_type, 'r') as load_f:
            encoding = json.load(load_f)
        encoding_data = []
        if self.encoding_type == "ProtVec":
            encoding_data = self.get_ProtVec_encoding(encoding, seq, overlap)
        else:
            for res in seq:
                if res not in SequenceEncoding.residue_types:
                    res = "X"
                encoding_data.append(encoding[res])

        return encoding_data

def ESM2_embedding_train_test(directory_path):
    """""
    Loads all .pt files from directory created by running the command that allows 
    the extraction of the final-layer embedding for a FASTA file from the ESM-2 model.  
    The command can be found under "Compute embeddings in bulk from FASTA" at 
    https://github.com/facebookresearch/esm. 0 padding is added to the dimension 
    representing the sequence length. Returns the ESM-2 embedding and binary labels 
    from the original FASTA file corresponding to the two classes.  
    """""
    x = [] 
    y = []

    #Load all .pt files from the ESM2 output  
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            ESM2_dict = torch.load(file_path)
            ESM2_last_layer = ESM2_dict['representations'][30]
            label = ESM2_dict['label']
        
            #0 padding to correct for varying sequence length  
            pad = nn.ConstantPad2d((0,0,(1000 - ESM2_last_layer.size(0)),0), 0)
            tmp_x = pad(ESM2_last_layer)
            x.append(tmp_x)
            
            #Save labels 
            if label.startswith('allergen'):
                y.append(1)
            else:
                y.append(0)
    
    x = torch.stack(x, dim=0)
    y = torch.tensor(y, dtype=torch.float)
    
    return (x, y)

def ESM2_embedding_predict(directory_path):
    """""
    Loads all .pt files from directory created by running the command that allows 
    the extraction of the final-layer embedding for a FASTA file from the ESM-2 model.  
    The command can be found under "Compute embeddings in bulk from FASTA" at 
    https://github.com/facebookresearch/esm. 0 padding is added to the dimension 
    representing the sequence length. Returns the ESM-2 embedding and protein ID's
    from the original FASTA file. 
    """""
    x = [] 
    y = []

    #Load all .pt files from the ESM2 output  
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            ESM2_dict = torch.load(file_path)
            ESM2_last_layer = ESM2_dict['representations'][30]
            label = ESM2_dict['label']
        
            #0 padding to correct for varying sequence length  
            pad = nn.ConstantPad2d((0,0,(1000 - ESM2_last_layer.size(0)),0), 0)
            tmp_x = pad(ESM2_last_layer)
            x.append(tmp_x)
            
            #Save labels 
            y.append(label)
    
    x = torch.stack(x, dim=0)
    y = np.array(y, dtype=object)
    
    return (x, y)

def format_and_encode_predict(predict_fasta, encoding_type, data_directory):
    """""
    Loads in FASTA file and encodes the sequences depending on the chosen encoding type.
    Returns the encoding and the protein ID's. data_directory is the directory path to 
    the .json files for the encoding.
    """""
    formatfasta = []
    recordid = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        fastalist = []
        length = len(record.seq)
        if length <= 1000:
            for i in range(1, 1000-length+1):
                fastalist.append('X')
            for a in record.seq:
                fastalist.append(a.upper())
        formatfasta.append(fastalist)
        recordid.append(record.id)
    inputarray = np.array(formatfasta)
    idarray = np.array(recordid, dtype=object)

    #Get encoding and covert to tensor
    seqEncoding = SequenceEncoding(encoding_type)
    x = []
    for i in range(len(inputarray)):
        x.append(seqEncoding.get_encoding(inputarray[i], data_directory))

    x = torch.tensor(x, dtype=torch.float)
    y = idarray

    return (x, y)


def format_and_encode_train_test(predict_fasta, encoding_type, data_directory):
    """""
    Loads in FASTA file and encodes the sequences depending on the chosen encoding type.
    Returns the encoding and binary labels from the original FASTA file corresponding to 
    the two classes. data_directory is the directory path to the .json 
    files for the encoding. 
    """"" 
    #Load in fasta_file
    formatfasta = []
    recordlabel = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        fastalist = []
        length = len(record.seq)
        if length <= 1000:
            for i in range(1, 1000-length+1):
                fastalist.append('X')
            for a in record.seq:
                fastalist.append(a.upper())
        formatfasta.append(fastalist)
        if record.id.startswith('allergen'):
            recordlabel.append(1)
        else:
            recordlabel.append(0)
    inputarray = np.array(formatfasta)
    labelarray = np.array(recordlabel)
        
    #Get encoding and covert to tensor
    seqEncoding = SequenceEncoding(encoding_type)
    x = []
    for i in range(len(inputarray)):
        x.append(seqEncoding.get_encoding(inputarray[i], data_directory))

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(labelarray, dtype=torch.float)

    return (x, y)

def num_of_input_channels(encoding_type):
    
    three_dim = ['AESNN3']
    four_dim = ['ANN4D']
    five_dim = ['Binary_5_bit','Acthely_factors']
    six_dim = ['One_hot_6_bit',]
    seven_dim = ['Meiler_parameters'] 
    twenty_dim = ['Hydrophobicity_matrix','PAM250','BLOSUM62','Miyazawa_energies','Micheletti_potentials']
    twentyone_dim = ['One_hot']
    hundered_dim = ['ProtVec']
     
    
    if encoding_type in three_dim:
        n = 3
    elif encoding_type in four_dim:
        n = 4   
    elif encoding_type in five_dim:
        n = 5
    elif encoding_type in six_dim:
        n = 6
    elif encoding_type in seven_dim:
        n = 7
    elif encoding_type in twenty_dim:
        n = 20
    elif encoding_type in twentyone_dim:
        n = 21  
    elif encoding_type in hundered_dim:
        n = 100
    elif encoding_type == 'ESM2':
        n = 640    
    return n 
    
