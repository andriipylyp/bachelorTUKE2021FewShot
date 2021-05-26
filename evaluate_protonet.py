import os
import torch
from tqdm import tqdm
import logging
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from libs.mini_objecta_dataLoader import FSDataLoader
from libs.models.protonet_model import PrototypicalNetwork

def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def train(i):
    classes_num = 5
    

    dataloader = FSDataLoader()

    loss = nn.CrossEntropyLoss()
    model = PrototypicalNetwork(3,
                                out_channels=5,
                                hidden_size=84)
    model.load_state_dict(torch.load('trained parameters/protonet_miniimagenet_5shot_5way.pt'))
    model.to(device='cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tasks = 100
    accuracy_l = []
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

            prototypes = get_prototypes(train_embeddings, train_targets,
                classes_num)
            loss = prototypical_loss(prototypes, test_embeddings, test_targets)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                accuracy_l.append(accuracy.item())
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            if batch_idx >= tasks:
                break

    # Save model
    # filename = os.path.join('trained parameters', 'protonet_miniobjectnet_'
    #     '{0}shot_{1}way.pt'.format(5, 5))
    # with open(filename, 'wb') as f:
    #     state_dict = model.state_dict()
    #     torch.save(state_dict, f)
    if i == 9:
        plt.xlabel('Tasks (1 epoch)')
        plt.ylabel('Accuracy')
        plt.title('Protonet miniobjectnet training (100 tasks)')
        plt.plot(accuracy_l)
        plt.show()
    return sum(accuracy_l) / len(accuracy_l)


if __name__ == '__main__':
    for i in range(10):
        print(train(i))