import torch
import random
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from problem import Problem
import make_labels




def read_section(file, marker=''):
    content = str()
    while line := file.readline():
        if marker in line:
            return content
        else:
            content += line


def process_solutions(filename, problem_map):
    with open(filename) as f:
        while line := f.readline():
            if "~~~START_SOLUTION~~~" in line:
                continue
            else:
                number = int(line)
                solution = read_section(f, "~~~END_SOLUTION~~~")
                problem_map[number].solutions.append(solution)



def get_dataset():
    statements_path = '/Users/victor/College/Fall-Quarter/Research/Feature-Classifier/src/clean/statements_cleaned1.txt'
    solutions_path = '/Users/victor/College/Fall-Quarter/Research/Feature-Classifier/src/clean/solutions_cleaned1.txt'

    statements_file = open(statements_path, 'r')
    raw_data = statements_file.readlines()
    raw_statements = [statement_num_pair.split('='*10 + '>')[1].replace('\n', '') for statement_num_pair in raw_data]
    problem_list = [Problem(int(statement_num_pair.split('='*10 + '>')[0]), statement_num_pair.split('='*10 + '>')[1].replace('\n', '')) for statement_num_pair in raw_data]
    problem_map = {}

    random.seed(a=12)
    random.shuffle(problem_list)

    for problem in problem_list:
        problem_map[problem.number] = problem

    process_solutions(solutions_path, problem_map)

    # for i in range(10):
    #     print(problem_list[i].number)
    #     print(problem_list[i].statement)
    #     for s in problem_list[i].solutions:
    #         print(s)
    #     print('='*100)


    statements_list = [prob.statement for prob in problem_list]
    label_list = make_labels.maps(problem_list)

    num_samples = len(statements_list)
    train_fraction = 0.75

    train_statements = statements_list[:int(num_samples * train_fraction)]
    train_labels = label_list[:int(num_samples * train_fraction)]

    val_statements = statements_list[int(num_samples * train_fraction):]
    val_labels = label_list[int(num_samples * train_fraction):]

    # print(train_statements[1])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_statements, truncation=True, padding='max_length', max_length=512)
    val_encodings = tokenizer(val_statements, truncation=True, padding='max_length', max_length=512)

    return train_encodings, train_labels, val_encodings, val_labels

    
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
    
