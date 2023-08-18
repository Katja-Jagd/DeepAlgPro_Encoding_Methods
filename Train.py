import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
import torchmetrics
import torch.optim as optim
from model import convATTnet
from logger import Logger
import time
from sklearn.model_selection import KFold
logger = Logger()
from Preprocessing import format_and_encode_train_test
from Preprocessing import SequenceEncoding  
from Preprocessing import ESM2_embedding_train_test
from Preprocessing import num_of_input_channels

def log(str, log_out):
    print(str)
    logger.set_filename(log_out)
    logger.log(str + '\n')


if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def validation(model, x_valid, y_valid_label, criterion, args):
    valid_ids = TensorDataset(x_valid, y_valid_label)
    valid_loader = DataLoader(
        dataset=valid_ids, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model.eval()
    accuracy = torchmetrics.Accuracy().to(device)
    recall = torchmetrics.Recall(average='micro').to(device)
    precision = torchmetrics.Precision().to(device)
    auroc = torchmetrics.AUROC(num_classes=None, average='micro').to(device)
    f1 = torchmetrics.F1Score().to(device)

    finaloutputs = torch.tensor([]).to(device)
    finallabels = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        valid_loss = 0.0
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
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
    return (valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value)


def train(args):
    if args.et == 'ESM2':
        x, y = ESM2_embedding_train_test(args.inputs.replace('.fasta','') + '_emb_esm2')
    else:
        x, y = format_and_encode_train_test(args.inputs, args.et, args.ed) 
    valid_loss_sum, accuracy_sum, recall_sum, precision_sum, auroc_sum, f1_sum = 0, 0, 0, 0, 0, 0
    k = 0
    skf = KFold(n_splits=10, shuffle=False)
    for train_index, valid_index in skf.split(x):
        x_train, x_valid = x[train_index], x[valid_index]
        y_train_label, y_valid_label = y[train_index], y[valid_index]
        k = k+1
        model = convATTnet(num_of_input_channels(args.et))       ### ADDED INPUT ARGUMENT
        model.to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
        criterion = nn.BCELoss().to(device)
        train_ids = TensorDataset(x_train, y_train_label)
        train_loader = DataLoader(
            dataset=train_ids, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        model.train()
        best_f1 = 0
        for epoch in range(args.epochs):
            train_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                log('[k: %d, batch: %d] train_loss: %.3f' %
                    (k, i + 1, loss.item()), 'train.log')
                now = time.asctime(time.localtime(time.time()))
            valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value = validation(
                model, x_valid, y_valid_label, criterion, args)
            # Generate validation results for each epoch.
            log('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (epoch, valid_loss, accuracy_value, recall_value,
                precision_value, auroc_value, f1_value), './model.'+str(k)+'.fold.everyepoch.valid.txt')
            # Save the model with the largest F1 value.
            if f1_value > best_f1:
                best_f1 = f1_value
                torch.save(model.state_dict(), './model.'+str(k)+'.pt')
                valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value = validation(
                    model, x_valid, y_valid_label, criterion, args)
        # Generate validation results for each fold
        log('[k: %d] valid_loss: %.3f accuracy_value: %.6f recall_value: %.6f precision_value: %.6f auroc_value: %.6f f1_value: %.6f' % (
            k, valid_loss, accuracy_value, recall_value, precision_value, auroc_value, f1_value), 'valid.log')
        valid_loss_sum += valid_loss
        accuracy_sum += accuracy_value.item()
        recall_sum += recall_value.item()
        precision_sum += precision_value.item()
        auroc_sum += auroc_value.item()
        f1_sum += f1_value.item()
    log('valid_loss: %.3f accuracy_value: %.6f recall_value: %.6f precision_value: %.6f auroc_value: %.6f f1_value: %.6f' % (
        valid_loss_sum/10, accuracy_sum/10, recall_sum/10, precision_sum/10, auroc_sum/10, f1_sum/10), 'valid.log')
    now = time.asctime(time.localtime(time.time()))
    log(str(now), 'train.log')
