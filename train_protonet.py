import os
import torch
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt
from torchvision import transforms
from torchmeta.transforms import ClassSplitter, Categorical
from torchmeta.datasets.miniimagenet import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

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


def train(  ):
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor()
    ])
    batch_size = 1
    dataset_transform = ClassSplitter(shuffle=True, num_train_per_class=5, num_test_per_class=5)
    dataset = MiniImagenet('data', transform=transform, num_classes_per_task=5, target_transform=Categorical(num_classes=5) ,meta_split="train", dataset_transform=dataset_transform )

    dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PrototypicalNetwork(3,
                                out_channels=5,
                                hidden_size=84)
    model.to(device='cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tasks = 15000
    accuracy_l = []
    # Training loop
    with tqdm(dataloader, total=tasks) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device='cuda')
            train_targets = train_targets.to(device='cuda')
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device='cuda')
            test_targets = test_targets.to(device='cuda')
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets,
                dataset.num_classes_per_task)
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
    filename = os.path.join('trained parameters', 'protonet_miniimagenet_'
        '{0}shot_{1}way.pt'.format(5, 5))
    with open(filename, 'wb') as f:
        state_dict = model.state_dict()
        torch.save(state_dict, f)
    plt.xlabel('Tasks (1 epoch)')
    plt.ylabel('Accuracy')
    plt.title('Protonet miniimagenet training (15k tasks)')
    plt.plot(accuracy_l[::150])
    plt.show()
    print(sum(accuracy_l) / len(accuracy_l))


if __name__ == '__main__':
    train()