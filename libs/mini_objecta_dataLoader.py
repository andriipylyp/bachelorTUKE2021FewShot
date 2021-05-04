import torch
from skimage import io
import pandas as pd
import random
from libs.mini_objecta_helper import dataGenerator

class FSDataLoader:
    def __init__(self, create_miniObjecta = False, num_ways = -1, num_shots = -1, num_tasks = -1):
        self.dataset = miniObjectNetDataloader(num_ways = num_ways, num_shots = num_shots, num_tasks = num_tasks, create_miniObjecta=create_miniObjecta)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, task=None):
        self.data = {}
        if task == None or task >= self.dataset.tasks:
            task = random.randint(0, self.dataset.tasks-1)
        self.dataset.task = task
        for stage in ['Train', 'Test']:
            data, labels = self.dataset[stage]
            self.data[stage] = (torch.tensor([data]), torch.tensor([labels]))
        return self.data

class miniObjectNetDataloader:
    def __init__(self, num_ways, num_shots, num_tasks, create_miniObjecta):
        'Initialization'
        self.dataGenerator = dataGenerator(num_ways, num_shots, num_tasks)
        if create_miniObjecta:
            self.dataGenerator.generateLabels()
        self.map = self.dataGenerator.loadMap()
        if self.map is None:
            raise ValueError('No map content, try create_miniObjecta=True')
        self.tasks = len(self.map)
        self.ways = len(self.map[0])
        self.shots = len(self.map[0][0]['train'])
        self.task = None

    def __len__(self):
        'Denotes the total number of samples'
        return self.ways*self.shots
    
    def __getitem__(self, stage):
        data = []
        label = []
        for way in range(len(self.map[self.task])):
            for shot in self.map[self.task][way]['train' if stage == 'Train' else 'test']:
                data.append(torch.tensor(self.dataGenerator.getImage(self.task, shot)).permute(2, 0, 1).numpy().tolist())
                label.append(way)
        return (data, label)
        

if __name__ == '__main__':
    dataloader = FSDataLoader()
    print(dataloader[None]['Test'].shape)