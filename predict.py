from model import convATTnet
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import os
from Preprocessing import format_and_encode_predict
from Preprocessing import SequenceEncoding  
from Preprocessing import ESM2_embedding_predict
from Preprocessing import num_of_input_channels

def main():
    argparser = argparse.ArgumentParser(
        description="DeepAlgPro Network for predicting allergens.")
    argparser.add_argument('-i', '--inputs', default='./',
                           type=str, help='input file')
    argparser.add_argument('-b', '--batch-size', default=1, type=int,
                           metavar='N')
    argparser.add_argument(
        '-o', '--output', default='allergenic_predict.txt', type=str, help='output file')
    argparser.add_argument('-et','--encoding-type', default="One_hot", type=str,
                           choices=['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix', 
                      'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies', 
                      'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec', 'ESM2'], dest='et')
    argparser.add_argument('-ed' '--encoding-directory', default='./', type=str, dest='ed')
    
    args = argparser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    predict(args)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("We will use "+torch.cuda.get_device_name())
else:
    device = torch.device('cpu')

class MyDataset(Dataset):
    def __init__(self, sequence, labels):
        self._data = sequence
        self._label = labels

    def __getitem__(self, idx):
        sequence = self._data[idx]
        label = self._label[idx]
        return sequence, label

    def __len__(self):
        return len(self._data)


def predict(args):

    if args.et == 'ESM2':
        profasta, proid = ESM2_embedding_predict(args.inputs.replace('.fasta','') + '_emb_esm2')
    else:
        profasta, proid = format_and_encode_predict(args.inputs, args.et, args.ed) 

    data_ids = MyDataset(profasta, proid)
    data_loader = DataLoader(
        dataset=data_ids, batch_size=args.batch_size, shuffle=False)

    # load the model
    model = convATTnet(num_of_input_channels(args.et))
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model.'+ args.et +'.pt'), strict=True)
    else:
        model.load_state_dict(torch.load('model.'+ args.et +'.pt', map_location=torch.device('cpu')),
                              strict=True)
    model.eval()
    with torch.no_grad():
        pred_r = []
        for i, data in enumerate(data_loader, 0):
            inputs, inputs_id = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            if device == torch.device('cpu'):
                probability = outputs[0].item()
            else:
                probability = outputs.item()
            if probability > 0.5:
                pred_r.append(
                    [''.join(inputs_id), probability, 'allergenicity'])
            else:
                pred_r.append(
                    [''.join(inputs_id), probability, 'non-allergenicity'])
    # generate outfile file
    df = pd.DataFrame(pred_r, columns=['protein', 'scores', 'predict result'])
    df.to_csv(args.output, sep='\t', header=True, index=True)


if __name__ == '__main__':
    main()
