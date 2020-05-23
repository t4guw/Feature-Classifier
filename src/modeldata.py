import random

class ModelData:
    def __init__(self):
        self.training_embeddings = []
        self.training_labels = []

        self.validation_embeddings = []
        self.validation_labels = []
    
    def add_data(self, embedding, label):
        if random.random() <= 0.7 :
			training_embeddings.append(embedding)
			training_labels.append(label)
        else:
            validation_embeddings.append(embedding)
            validation_labels.append(label)


