import torch
import torch.nn.functional as func
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from pathlib import Path
from get_dataset import get_dataset
import random

random.seed(20)

class LeetcodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings, train_labels, val_encodings, val_labels = get_dataset()

train_dataset = LeetcodeDataset(train_encodings, train_labels)
val_dataset = LeetcodeDataset(val_encodings, val_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=768, out_features=2048),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=2048, out_features=2),
    torch.nn.Softmax(dim=1),
)
print(model)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

accuracy_list = []




def validate(acc_list):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            print(total + 1)
            print("LABEL:\n", label[0])
            output = model(input_ids, attention_mask=attention_mask, labels=label)[1]
            print("OUTPUT:\n", output)
            pred = 0
            if output[0][1] > 0.5:
                pred = 1
            
            if pred == label[0]:
                correct += 1
            total += 1
            
    print('ACCURACY:', correct / total)
    acc_list.append(correct / total)
    return acc_list



accuracy_list = validate(accuracy_list)
epoch_num = 10

if True:
    print('starting training\n')
    for epoch in range(epoch_num):
        model.train()
        print('EPOCH', epoch + 1)
        batch_num = 0
        loss_sum = 0
        for batch in train_loader:
            print('  batch', batch_num)
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss_sum += loss

            loss.backward()
            optim.step()
            batch_num += 1
            
        print(loss_sum / batch_num)
        accuracy_list = validate(accuracy_list)

    print('LOSS LIST')
    print(loss_list)
    print('ACCURACY LIST')
    print(accuracy_list)
    print()          
    print('finished training')