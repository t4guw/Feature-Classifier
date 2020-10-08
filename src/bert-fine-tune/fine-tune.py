import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from pathlib import Path
from get_dataset import get_dataset


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
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)


model.eval()
# print(model)      
with torch.no_grad():
    # Forward pass, calculate logit predictions
    output = model(torch.LongTensor(train_encodings.input_ids[:3]), \
    attention_mask=torch.LongTensor(train_encodings.attention_mask[:3]), \
    labels=torch.LongTensor(train_labels[:3]))
    print(output)

# exit(0)

print('starting training\n')
for epoch in range(3):
    print('EPOCH', epoch + 1)
    batch_num = 1
    for batch in train_loader:
        print('  batch', batch_num)
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        print(input_ids.size())
        print()
        exit(0)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        batch_num += 1

print('finished training')
model.eval()
print('done')



# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
# bert_model.train() # put into train mode


# optimizer = AdamW(optimizer_grouped_params, lr=1e-5)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# text_batch = ['this thing is amazing!!', 'this thing is horrible...']
# encoding  = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
# input_ids = encoding['input_ids']
# attention_mask = encoding['attention_mask']

# labels = torch.tensor([0, 1]).unsqueeze(0)
# outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
# loss = outputs.loss
# loss.backward()
# optimizer.step()

