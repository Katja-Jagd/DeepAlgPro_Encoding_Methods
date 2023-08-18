import torch
import torchmetrics
from model import convATTnet
from logger import Logger
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
from Preprocessing import format_and_encode_train_test
from Preprocessing import SequenceEncoding  
from Preprocessing import ESM2_embedding_train_test
from Preprocessing import num_of_input_channels

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

logger = Logger()


def log(str, log_out):
    print(str)
    logger.set_filename(log_out)
    logger.log(str + '\n')

def test(args):
    if args.et == 'ESM2':
        x, y = ESM2_embedding_train_test(args.inputs.replace('.fasta','') + '_emb_esm2')
    else:
        x, y = format_and_encode_train_test(args.inputs, args.et, args.ed) 

    test_ids = TensorDataset(x, y)
    test_loader = DataLoader(
        dataset=test_ids, batch_size=args.batch_size, shuffle=True)
    model = convATTnet(num_of_input_channels(args.et))    
    model.to(device)
    model.load_state_dict(torch.load('./model.'+ args.et +'.pt',map_location='cuda:0' ), strict=True)
    model.eval()
    accuracy = torchmetrics.Accuracy().to(device)
    recall = torchmetrics.Recall(average='micro').to(device)
    precision = torchmetrics.Precision().to(device)
    auroc = torchmetrics.AUROC(num_classes=None, average='micro').to(device)
    f1 = torchmetrics.F1Score().to(device)
    finaloutputs = torch.tensor([]).to(device)
    finallabels = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            labels = torch.as_tensor(labels, dtype=torch.long)
            finaloutputs = torch.cat([finaloutputs, outputs], 0)
            finallabels = torch.cat([finallabels, labels], 0)
        accuracy(finaloutputs, finallabels)
        recall(finaloutputs, finallabels)
        precision(finaloutputs, finallabels)
        auroc(finaloutputs, finallabels)
        f1(finaloutputs, finallabels)
        accuracy_value = accuracy.compute()
        recall_value = recall.compute()
        precision_value = precision.compute()
        auroc_value = auroc.compute()
        f1_value = f1.compute()
        accuracy.reset()
        recall.reset()
        precision.reset()
        auroc.reset()
        f1.reset()
        log('Test Result: F1: ' + str("%.5f" % f1_value.item()) + '\tAccurcay: ' + str("%.5f" % accuracy_value.item()) + '\tPrecision: ' + str("%.5f" %
            precision_value.item()) + '\tRecall: ' + str("%.5f" % recall_value.item())+'\tAUROC: ' + str("%.5f" % auroc_value.item()), './test.log')
