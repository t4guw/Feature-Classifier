from transformers import BertTokenizer

# input_ids
# token_type_ids
# attention_mask

train_texts, train_labels = ["yay this is a happy text!", "grrrr this is a mad text.", "hmmm... this is neutral."], [1, 0, 1]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

print(train_encodings)

