from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np

def get_train_idxs(dataset, num_users, num_items, alpha):
    labels = dataset.targets
    
    # Collect idxs for each label
    idxs_labels = {i: set() for i in range(10)}
    for idx, label in enumerate(labels):
        idxs_labels[label].add(idx)
    

    # 10 labels
    class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    class_dist = (class_dist * num_items).astype(int)
    
    if num_users == 1:
        for _class, class_num in enumerate(class_dist[0]):
            if class_num > len(idxs_labels[_class]):
                class_dist[0][_class] = len(idxs_labels[_class])
            
    else:   
        for _class, class_num in enumerate(class_dist.T.sum(axis=1)):
            assert class_num < len(idxs_labels[_class]), "num_items must be smaller"
    
    
    dict_users = {i: set() for i in range(num_users)}
    dists = {i: [0 for _ in range(10)] for i in range(num_users)}
    
    for client_id, client_dist in enumerate(class_dist):
        for _class, num in enumerate(client_dist):
            sample_idxs = idxs_labels[_class]
            dists[client_id][_class] += num
            
            sampled_idxs = set(np.random.choice(list(sample_idxs), size=num, replace=False)) 
            # accumulate
            dict_users[client_id].update(sampled_idxs)
            
            # exclude assigned idxs
            idxs_labels[_class] = sample_idxs - sampled_idxs
            
    for i, data_idxs in dict_users.items():
        dict_users[i] = list(data_idxs)
    
    server_data_idx = {i: list(idxs) for i, idxs in idxs_labels.items()}

    return dict_users, server_data_idx


class SimCLRTransformWrapper(object):
    def __init__(self, args):
        self.base_transform = SimCLRTransform(args)
        self.n_views = args.n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)] # two views by default

def SimCLRTransform(args):
    s = args.strength
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def NoTransform(args):
    return transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def get_dataset(args):
    cifar_data_path = os.path.join(args.data_path, "cifar")
    mnist_data_path = os.path.join(args.data_path, "mnist")
    if args.dataset == "cifar":
        data = datasets.CIFAR10
        data_path = cifar_data_path

    elif args.dataset == "mnist":
        data = datasets.MNIST
        data_path = mnist_data_path

    # # transforms set according to 
    # # https://github.com/guobbin/PFL-MoE/blob/master/main_fed.py
    # simclr_transform = SimCLRTransform(args)
    # no_transform = NoTransform(args)

    train_dataset = data(
        data_path, 
        train=True,  
        download=True
    )

    test_dataset = data(
        data_path, 
        train=False,
        download=True,
        transform=NoTransform(args)
    )

    user_train_idxs, server_data_idx = get_train_idxs(
        train_dataset, 
        args.num_users, 
        args.num_items,
        args.alpha
    )

    train_custom_sets = {client_id: CustomData(args, idxs, train_dataset) for client_id, idxs in user_train_idxs.items()}
    # server_dataset = {class_idx: CustomData(args, idxs, train=True, download=True) for class_idx, idxs in server_data_idx.items()}
    return train_custom_sets, server_data_idx, train_dataset, test_dataset
    
    #** 
def get_loader(args, shuffle, batch_size, transform, dataset):
    dataset.transform = transform
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last = True
    )
    return loader

class CustomData(Dataset):
    def __init__(self, 
            args, 
            idxs,
            dataset
        ):
        super(CustomData, self).__init__()
        path = os.path.join(args.data_path, args.dataset)
        # if args.dataset == "cifar":
        #     data = datasets.CIFAR10(path, train = train, download = download)
        # elif args.dataset == "mnist":
        #     data = datasets.MNIST(path, train = train, download = download)

        self.subset = Subset(
            dataset, 
            idxs
        )

        self.N = len(idxs)

        self.transform = None


    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        assert self.transform != None, "transform must be set"
        return (self.transform(self.subset[index][0]), self.subset[index][1])

    
    
