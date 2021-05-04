import os
import pandas as pd
from shutil import copyfile, Error
import shutil
import glob
import random
import json
import ast
from PIL import Image
from skimage import io, color

class dataGenerator:
    def __init__(self,num_ways=-1,num_shots=-1,num_tasks=-1,path_to_objectnet='',path_to_mini_dataset=''):
        self.path = path_to_mini_dataset
        if self.path == '':
            self.path = 'data/'
        self.path_to_objectnet = path_to_objectnet
        self.generateDirStructures()
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.path_to_objectnet = path_to_objectnet
        self.num_tasks = num_tasks
        self.map_structure = {}

    def generateLabels(self):
        if self.num_ways == -1 or self.num_shots == -1 or self.num_tasks == -1:
            raise ValueError('num_ways or num_shots or num_tasks not defined...')
        folder = f'{self.path}miniObjecta\data'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        class_list = []
        images = []
        for _, dirnames, _ in os.walk(self.path_to_objectnet+'objectnet-1.0\objectnet-1.0\images'):
            for name in dirnames:
                class_list.append(name)
                image_buf = []
                for _, _, filenames in os.walk(self.path_to_objectnet+'objectnet-1.0\objectnet-1.0\images\{0}'.format(name)):
                    for filename in filenames:
                        image_buf.append(filename)
                    images.append(image_buf)
        for i in range(self.num_tasks):
            data = {}            
            for _ in range(self.num_ways):
                idx = random.randint(0,len(class_list)-1)
                c = class_list[idx]
                data[c] = {}
                for stage in ['test', 'train', 'evaluate']:
                    data[c][stage] = []
                    for _ in range(self.num_shots):
                        data[c][stage].append(images[idx][random.randint(0,len(images[idx])-1)])
            
            self.generateDataset(i, data)
        self.saveMap(self.map_structure)

    def loadMap(self):
            df = pd.read_csv(self.path+'miniObjecta/map.csv')
            dict_data = {}
            for column in df:
                dict_data[int(column)] = []
                for row in range(len(df[column])):
                    if df[column][row] == df[column][row]:
                        dict_data[int(column)].append(json.loads(df[column][row].replace("'", '"')))
            return dict_data

    def saveMap(self, map_s):
        df = pd.DataFrame(map_s)
        df.to_csv(self.path+'miniObjecta/map.csv', index=False)

    def generateDataset(self, task_num, data):
        idx = 0
        self.map_structure[task_num] = {}
        os.mkdir(f'{self.path}miniObjecta/data/{task_num}')
        for c in data:
            self.map_structure[task_num][c] = {}
            for stage in ['test', 'train', 'evaluate']:
                self.map_structure[task_num][c][stage] = []
                for image in data[c][stage]:
                    idx += 1
                    image = Image.open(f'{self.path_to_objectnet}objectnet-1.0/objectnet-1.0/images/{c}/{image}')
                    image = image.resize((84,84))
                    image.save(f'{self.path}miniObjecta/data/{task_num}/{idx}.png')
                    self.map_structure[task_num][c][stage].append(f'{idx}.png')

    def getImage(self, task_num, idx):
        if not os.path.isdir(self.path_to_objectnet+'miniObjecta'):
            if self.path == '':
                self.path = 'data/'
        return color.rgba2rgb(io.imread(f'{self.path}miniObjecta/data/{task_num}/{idx}') / 255)

    def generateDirStructures(self):
        if not os.path.isdir('data/miniObjecta'):
            if self.path == '':
                self.path = 'data/'
            os.mkdir(self.path+'miniObjecta')
            if not os.path.isdir(self.path+'miniObjecta/data'):
                os.mkdir(self.path+'miniObjecta/data')

if __name__ == "__main__":
    generator = dataGenerator(5,5,num_tasks=100)
    generator.generateLabels()
    # print(generator.loadMap()[0])