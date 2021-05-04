# from mini_imagenet_dataloader import MiniImageNetDataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchmeta.utils.gradient_based import gradient_update_parameters
from libs.models.maml_model import MetaConvModel
from libs.mini_objecta_dataLoader import FSDataLoader

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def ModelConvMiniImagenet(out_features, hidden_size=84):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

if __name__ == "__main__":
    classes_num = 5
    model = ModelConvMiniImagenet(classes_num)
    model.load_state_dict(torch.load('trained parameters/maml_miniImagenet_5shot_5way_15000t.th'))
    model.zero_grad()

    dataloader = FSDataLoader()

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy_l = list()
    loss = nn.CrossEntropyLoss()
    model.train()
    num_of_tasks = 100
    epochs = 2
    with tqdm(dataloader, total=num_of_tasks) as qbar:
        for idx, batch in enumerate(qbar):
            model.zero_grad()
            train_inputs, train_targets = batch['Train']
            test_inputs, test_targets = batch['Test']
            
            for _ in range(epochs):
                outer_loss = torch.tensor(0., device='cpu')
                accuracy = torch.tensor(0., device='cpu')
                train_logit = model(train_inputs)
                inner_loss = F.cross_entropy(train_logit, train_targets)
                  
                params = gradient_update_parameters(model, inner_loss)

                test_logit = model(test_inputs , params=params)
                outer_loss += F.cross_entropy(test_logit, test_targets)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_targets)
                outer_loss.div_(1)
                accuracy.div_(1)

                outer_loss.backward()
                meta_optimizer.step()
            accuracy_l.append(accuracy.item())
            if idx > num_of_tasks-1:
                break
    plt.title('MAMAL meta-training on 100 tasks')
    plt.xlabel('tasks (2 epoch)')
    plt.ylabel('accuracy')
    plt.plot(accuracy_l)
    plt.show()
    accuracy_l.clear()
