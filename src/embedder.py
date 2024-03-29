import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


# Converts a statement string into a list of tokens readable by BERT
# Returns this list of tokens
def get_tokenized_statement(statement, tokenizer):
    tok_sentence = tokenizer.tokenize('[CLS] ' + statement + ' [SEP]')
    if len(tok_sentence) > 512:
        tok_sentence = tok_sentence[-511:]
        tok_sentence.insert(0, '[CLS]')

    return tok_sentence


# Returns the same list of tokens as get_tokenized_statement() above,
# except it maps each token string (e.g. "cat") to its index in BERT's
# vocabulary (e.g. 251)
def get_indexed_tokens(tok_sentence, tokenizer):
    return tokenizer.convert_tokens_to_ids(tok_sentence)


# Extracts embedding from the BERT network's output layers
# Returns the embedding
def extract_embeddings(encoded_layers):
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs = encoded_layers[11][0]
    statement_embedding = torch.mean(token_vecs, dim=0)

    return statement_embedding


# Iterates through all statement strings and creates an embedding of each one
# Each embedding is a vector (type = torch tensor) of 786 dimensions
# Reterns a list of all embeddings
def get_embeddings(all_problems):
    print('Go BERT, go!!')

    output = open('embeddings.txt', 'w')

    all_embeddings = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    index = 1
    
    for problem in all_problems:
        tok_statement = get_tokenized_statement(problem['statement'], tokenizer)
        indexed_tokens = get_indexed_tokens(tok_statement, tokenizer)
        segment_ids = [1] * len(tok_statement)
        
        tok_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segment_ids])

        with torch.no_grad():
            encoded_layers, x = bert_model(tok_tensor, segments_tensor)

        statement_embedding = extract_embeddings(encoded_layers)
        all_embeddings.append(statement_embedding.tolist())
        for num in all_embeddings[-1]:
            output.write(str(num))
            output.write(' ')
        output.write('\n')
        output.flush()
        print(index, '/ 762')
        index += 1
    
    print(all_embeddings[0], all_embeddings[5], all_embeddings[762])
    
    print('DONE: Generated all BERT embeddings')
    return all_embeddings


def get_embeddings_from_file():
    emb_file = open('embeddings.txt', 'r')
    all_embeddings = []
    for line in emb_file.readlines():
        vector = [float(num) for num in line.replace(' \n', '').split(' ')]
        all_embeddings.append(vector)

    return all_embeddings