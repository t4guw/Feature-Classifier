import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from pathlib import Path

print(BertForSequenceClassification.from_pretrained('bert-base-uncased'))
exit(0)
print('start')

# try making new environment with python 3.7 instead of 3.8
abs_path = '/Users/victor/College/Fall-Quarter/Research/Feature-Classifier/src/bert-fine-tune/'

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

print('creating dataset...')
train_texts, train_labels = read_imdb_split(abs_path + 'aclImdb/train')
test_texts, test_labels = read_imdb_split(abs_path + 'aclImdb/test')
print('created dataset')
print('train length:', len(train_texts), '\tnum batches:', len(train_texts) / 16)

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
print('partitioned dataset')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
print('created encodings')

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)
print('created final dataset')

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# model.to(device)
model.train()
print('initialized bert model')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

print('starting training\n')
for epoch in range(3):
    print('EPOCH', epoch + 1)
    batch_num = 1
    for batch in train_loader:
        print('  batch', batch_num)
        optim.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
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

