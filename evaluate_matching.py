import os
import torch
from tqdm import tqdm
from torchvision import transforms
from torchmeta.transforms import ClassSplitter, Categorical
from torchmeta.datasets.miniimagenet import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.matching import matching_log_probas, matching_loss
import matplotlib.pyplot as plt
from libs.mini_objecta_dataLoader import FSDataLoader
from libs.models.matching_model import MatchingNetwork
def train():

    num_ways = 5
    num_shots = 5
    tasks = 100
    dataloader = FSDataLoader()
    
    model = MatchingNetwork(3,out_channels=num_ways, hidden_size=84)
    model.load_state_dict(torch.load('trained parameters/matching_network_miniimagenet_5shot_5way.pt'))
    model.to(device='cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy_l = list()
    # Training loop
    with tqdm(dataloader, total=tasks) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['Train']
            train_inputs = train_inputs.to(device='cuda')
            train_targets = train_targets.to(device='cuda')
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['Test']
            test_inputs = test_inputs.to(device='cuda')
            test_targets = test_targets.to(device='cuda')
            test_embeddings = model(test_inputs)

            loss = matching_loss(train_embeddings,
                                 train_targets,
                                 test_embeddings,
                                 test_targets,
                                 num_ways)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # calculate the accuracy
                log_probas = matching_log_probas(train_embeddings,
                                                 train_targets,
                                                 test_embeddings,
                                                 num_ways)
                test_predictions = torch.argmax(log_probas, dim=1)
                accuracy = torch.mean((test_predictions == test_targets).float())
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
                accuracy_l.append(accuracy.item())

            if batch_idx >= tasks:
                break

    # Save model
    
    # filename = os.path.join('', 'matching_network_miniimagenet_'
    #     '{0}shot_{1}way.pt'.format(num_shots, num_ways))
    # with open(filename, 'wb') as f:
    #     state_dict = model.state_dict()
    #     torch.save(state_dict, f)
    plt.xlabel('Tasks (1 epoch)')
    plt.ylabel('Accuracy')
    plt.title('Matching network training miniObjecta 100 tasks')
    plt.plot(accuracy_l)
    plt.show()
    print(sum(accuracy_l) / len (accuracy_l))

if __name__ == '__main__':
    train()