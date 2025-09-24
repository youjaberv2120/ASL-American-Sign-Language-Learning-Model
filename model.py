# based on https://medium.com/@enrico.randellini/hands-on-video-classification-with-pytorchvideo-dc9cfcc1eb5f
# remember to set "export PYTORCH_ENABLE_MPS_FALLBACK=1" before running

import numpy as np
import csv
import os
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import Dataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.base_model.blocks[5].proj = nn.Sequential(
                                            nn.Linear(2048, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 128),
                                            nn.ReLU(),
                                            nn.Dropout(0.35),
                                            nn.Linear(128, 50))

    def forward(self, x):
        return self.base_model(x)

post_act = torch.nn.Softmax(dim=1)

def train_batch(inputs, labels, model, optimizer, criterion):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

@torch.no_grad()
def accuracy(inputs, labels, model):
    model.eval()
    outputs = model(inputs)
    preds = post_act(outputs)
    _, pred_classes = torch.max(preds, 1)
    is_correct = pred_classes == labels
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss_fn(inputs, labels, model, criterion):
    model.eval()
    outputs = model(inputs)
    val_loss = criterion(outputs, labels)
    return val_loss.item()

def run_model(min_lr, max_lr, fl_interval, patience, past_path, save_path):
    #load data
    label_dict = {} # gloss to label (numerical)
    with open('sample_classes.txt') as labels_file:
        content = labels_file.readlines()
        i = 0
        for line in content:
            label_dict[line[:-1]] = i
            i += 1

    splits = {} # each list: video ids in that split
    label_list = np.empty(0, dtype=int) # id to label
    id_to_filename = [] # id to filename
    video_info = [] # id to begin/end frames
    splits['train'] = np.empty(0, dtype=int)
    splits['val'] = np.empty(0, dtype=int)
    splits['test'] = np.empty(0, dtype=int)
    with open('augmented_mini_dataset.csv') as data:
        reader = csv.reader(data)
        next(reader)

        id = 0
        for line in reader:
            split = line[0]
            file_name = line[1]
            start_frame = int(line[2])
            end_frame = int(line[3])
            label = line[4]

            splits[split] = np.append(splits[split], id)
            label_list = np.append(label_list, label_dict[label])
            id_to_filename.append(file_name)
            video_info.append([start_frame, end_frame])
            id += 1

    #prepare for training
    train_dataset = Dataset(id_list=splits['train'], label_list=label_list, id_to_filename=id_to_filename, video_info=video_info)
    val_dataset = Dataset(id_list=splits['val'], label_list=label_list, id_to_filename=id_to_filename, video_info=video_info)
    test_dataset = Dataset(id_list=splits['test'], label_list=label_list, id_to_filename=id_to_filename, video_info=video_info)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, drop_last=False)

    device = torch.device('mps')

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, mode='triangular2', step_size_up=3500, step_size_down=3500)
    start_epoch = 0
    min_val_loss = 1000000
    min_val_loss_epoch = -1

    # loading past checkpoint
    if past_path:
        saved = torch.load(past_path, map_location=torch.device(device))
        start_epoch = saved['epoch'] + 1
        model.load_state_dict(saved['model'])
        model = model.to(device)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scheduler.step()
        min_val_loss = saved['best_loss']
        min_val_loss_epoch = saved['best_epoch']

    print(start_epoch + 1)
    print(min_val_loss, min_val_loss_epoch + 1)

    # freeze layers
    lowest_unfrozen_layer = max(5 - start_epoch // fl_interval, 2)
    for name, param in model.named_parameters():
        if int(name.split('.')[2]) < lowest_unfrozen_layer:
            param.requires_grad = False

    # train
    train_losses = np.empty(0, dtype=float)
    val_losses = np.empty(0, dtype=float)
    val_accs = np.empty(0, dtype=float)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for epoch in range(start_epoch, 40):
        # unfreeze layers
        if epoch % fl_interval == 0 and epoch <= 2 * fl_interval:
            layer_to_unfreeze = 5 - epoch/fl_interval
            for name, param in model.named_parameters():
                if int(name.split('.')[2]) == layer_to_unfreeze:
                    param.requires_grad = True

        # iterate on all train batches of the current epoch by executing the train_batch function
        train_curr_epoch_losses = []
        for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            train_curr_epoch_losses.append(train_batch(inputs, labels, model, optimizer, criterion))
        train_loss = round(np.array(train_curr_epoch_losses).mean(), 5)
        train_losses = np.append(train_losses, train_loss)

        # iterate on all batches of val of the current epoch by calculating the accuracy and the loss function
        val_curr_epoch_accuracies = []
        val_curr_epoch_losses = []
        for inputs, labels in tqdm(val_dataloader, desc=f'epoch {str(epoch + 1)} | val'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            val_curr_epoch_losses.append(val_loss_fn(inputs, labels, model, criterion))
            val_curr_epoch_accuracies.extend(accuracy(inputs, labels, model))
        val_loss = round(np.array(val_curr_epoch_losses).mean(), 5)
        val_acc = round(np.array(val_curr_epoch_accuracies).mean(), 5)
        val_losses = np.append(val_losses, val_loss)
        val_accs = np.append(val_accs, val_acc)

        print(train_loss)
        print(val_loss, val_acc)
        print(f'Time: {datetime.datetime.now()}')

        # logging
        if not os.path.isfile(os.path.join(save_path, 'losses.txt')):
            with open(os.path.join(save_path, 'losses.txt'), 'w') as ls_file:
                ls_file.write('epoch\ttrain loss\tval loss\tval acc\n')
        
        with open(os.path.join(save_path, 'losses.txt'), 'a') as ls_file:
            ls_file.write(f'{epoch + 1}\t\t{train_loss}\t\t{val_loss}\t\t{val_acc}\n')
            ls_file.close()

        # early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = epoch

            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': min_val_loss,
                'best_epoch': min_val_loss_epoch
            }
            torch.save(save_obj, os.path.join(save_path, 'best.pth'))
       
        if epoch - min_val_loss_epoch >= patience:
            print(f'Stopping early at epoch {epoch + 1}')
            break

        # save model
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': min_val_loss,
            'best_epoch': min_val_loss_epoch
        }
        torch.save(save_obj, os.path.join(save_path, f'{epoch + 1}.pth'))
        
        scheduler.step()
        torch.mps.empty_cache()
        print('---------------------------------------------------------')

    #test
    actual = []
    predicted = []
    total = 0
    best = torch.load(save_path + 'best.pth', map_location=torch.device(device))
    model.load_best_dict(saved['model'])
    model = model.to(device)
                     
    model = model.eval()
    print('Testing')
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc=f'test'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # Get the predicted classes
            preds = post_act(outputs)
            _, pred_classes = torch.max(preds, 1)
            actual.extend(labels.cpu().numpy().tolist())
            predicted.extend(pred_classes.cpu().numpy().tolist())
            numero_video = len(labels.cpu().numpy().tolist())
            total += numero_video

        # report predictions and true values to numpy array
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        print('Accuracy: ', accuracy_score(actual, predicted))
        print(metrics.classification_report(actual, predicted))

        # Plot confusion matrix
        cm = metrics.confusion_matrix(actual, predicted)

        fig, ax = plt.subplots(figsize=(50, 30))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=label_dict.keys(),
                yticklabels=label_dict.keys(), title="Confusion matrix")
        plt.yticks(rotation=0)
        fig.savefig(os.path.join(save_path, 'confusion_matrix.png'))

        # Save report in a txt
        target_names = list(label_dict.keys())
        cr = metrics.classification_report(actual, predicted, target_names=target_names)
        with open(os.path.join(save_path, 'report.txt'), 'w') as report:
            report.write('Classification Report\n\n{}'.format(cr))
        report.close()