import random

class ModelData:
    def __init__(self, embeddings, labels):
        self.all_embeddings = embeddings
        self.all_labels = labels

        self.training_embeddings = []
        self.training_labels = []

        self.validation_embeddings = []
        self.validation_labels = []

        self.set_data()
    
    def set_data(self):
        for i in range(len(self.all_labels)):
            if random.random() <= 0.7 :
                training_embeddings.append(self.all_embeddings[i])
                training_labels.append(self.all_labels[i])
            else:
                validation_embeddings.append(self.all_embeddings[i])
                validation_labels.append(self.all_labels[i])


