import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from pathlib import Path
from get_dataset import get_dataset

print(BertForSequenceClassification.from_pretrained('bert-base-uncased'))

print('start')


abs_path = '/Users/victor/College/Fall-Quarter/Research/'

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
print('train length:', len(train_texts), '\tnum batches:', len(train_texts) / 5)

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
print('partitioned dataset')

print(train_texts[0])
print('='*100)
print(train_texts[1])
print(type(train_texts[0]))

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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)
model.train()
print('initialized bert model')

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

print('starting training\n')
for epoch in range(3):
    print('EPOCH', epoch + 1)
    batch_num = 1
    for batch in train_loader:
        print('  batch', batch_num)
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        batch_num += 1

print('finished training')
model.eval()
