import random

class ModelData:
    def __init__(self, embeddings, labels):
        self.all_embeddings = list(embeddings)
        self.all_labels = list(labels)

        self.training_embeddings = []
        self.training_labels = []

        self.validation_embeddings = []
        self.validation_labels = []
    
    def set_data_7030(self):
        for i in range(len(self.all_labels)):
            if random.random() <= 0.7 :
                self.training_embeddings.append(self.all_embeddings[i])
                self.training_labels.append(self.all_labels[i])
            else:
                self.validation_embeddings.append(self.all_embeddings[i])
                self.validation_labels.append(self.all_labels[i])
    
    def set_data_kfold(self, k, index):
        remove_index = int(len(self.all_labels) / k * index)
        
        for i in range(int(len(self.all_labels) / k)):
            self.validation_embeddings.append(self.all_embeddings.pop(remove_index))
            self.validation_labels.append(self.all_labels.pop(remove_index))
        
        self.training_embeddings = self.all_embeddings
        self.training_labels = self.all_labels
