import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy 
import numpy as np

from data_utils import get_loader
# from utils import AverageMeter
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

def train_server_model(args, trainset, device, transform, sup_model, epochs=1):
    assert transform != None
    loader = get_loader(
        args = args, 
        batch_size = args.server_batch_size,
        shuffle = True, 
        transform = transform, 
        dataset = trainset
    )
    model = copy.deepcopy(sup_model).to(device)
    model.set_mode("linear")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr
    )
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader): 
            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images)
            loss = FL_criterion(device)(preds, labels)

            loss.backward()
            optimizer.step()
            
    return model.state_dict()

    

def train_client_model(args, trainset, device, transform, sup_model = None, unsup_model = None, epochs=1, q=None, done=None):
    assert transform != None
    assert unsup_model != None

    

    loader = get_loader(
        args = args, 
        batch_size = args.local_bs,
        shuffle = False, 
        transform = transform, 
        dataset = trainset
    )

    client_model = copy.deepcopy(unsup_model).to(device)
    if args.exp == "FLSL":
        client_model.set_mode("linear")

    else:
        client_model.set_mode("train")

    
    if args.agg == "FedSSL":
        ref_model = copy.deepcopy(sup_model).to(device)
        ref_model.set_mode("train") # frozen -> train mode outputs latent vector
    
    # fedavg, fedprox must replace unsup_model's weights with sup ones  
    else:
        client_model.load_state_dict(copy.deepcopy(sup_model.to(device).state_dict()))


    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, client_model.parameters()),
        args.lr
    )

    fedprox_glob_model = copy.deepcopy(unsup_model).to(device)
    fedprox_glob_model.set_mode("train")

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            optimizer.zero_grad()
            if args.exp == "FLSL" or args.exp == "centralized":
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                preds = client_model(images)
                loss = FL_criterion(device)(preds, labels)

            elif args.exp == "simclr":
                images1, images2 = images
                images1 = images1.to(device, non_blocking=True)
                images2 = images2.to(device, non_blocking=True)
                z1, z2 = client_model(images1, images2)
                loss = NCE_loss(device, args.temperature, z1, z2)
     
            
            elif args.exp == "simsiam":
                images1, images2 = images
                images1 = images1.to(device, non_blocking=True)
                images2 = images2.to(device, non_blocking=True)

                p1, p2, z1, z2 = client_model(images1, images2) 
                loss = SimSiam_loss(device, p1, p2, z1.detach(), z2.detach())



            if args.agg == "fedprox":
                if epoch > 0:
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(client_model.parameters(), fedprox_glob_model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += args.mu / 2. * w_diff
            
            elif args.agg == "FedSSL":
                if args.exp == "simclr":
                    with torch.no_grad():
                        ref_feature1, ref_feature2 = ref_model(images1, images2)

                    #cosine_sim = SimSiam_criterion(device)
                    mse_loss = nn.MSELoss().to(device)
                    loss += (mse_loss(ref_feature1, z1) + mse_loss(ref_feature2, z2))

                elif args.exp == "simsiam":
                    with torch.no_grad():
                        _, _, ref_feature1, ref_feature2 = ref_model(images1, images2)

                    #cosine_sim = SimSiam_criterion(device)
                    mse_loss = nn.MSELoss().to(device)
                    loss += (mse_loss(ref_feature1, z1) + mse_loss(ref_feature2, z2))
            
            loss.backward()
            optimizer.step()


    state_dict = client_model.state_dict()
    
    # Multiprocessing Queue
    if q != None:
        # state_dict = {key: tensor.clone().share_memory_() for key, tensor in state_dict.items()}
        q.put(state_dict)
        done.wait()

    return state_dict

def test_server_model(args, testset, device, transform, sup_model):
    assert transform != None
    assert sup_model != None

    model = copy.deepcopy(sup_model).to(device)
    model.set_mode("linear") 

    loader = get_loader(
        args = args, 
        batch_size = len(testset),
        shuffle = False, 
        transform = transform, 
        dataset = testset
    )

    images, labels = next(iter(loader))
    
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(images)

        loss = FL_criterion(device)(preds, labels)
        loss_value = loss.item()

        _, top1_preds = torch.max(preds.data, -1)
        _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

        top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
        top5 = 0
        for label, pred in zip(labels, top5_preds):
            if label in pred:
                top5 += 1

        top5 /= labels.size(0)
        top5 *= 100

    return loss_value, top1, top5

def test_client_model(args, testset, device, transform, unsup_model, finetune=True, finetune_epochs = 5):
    assert transform != None
    assert unsup_model != None

    # if finetune, finetune epochs must be > 0 
    assert (not finetune) or (finetune_epochs > 0 and finetune)
    
    if finetune:
        # use half of the testset as dataset for finetuning
        finetune_idxs = np.random.choice(range(len(testset)), len(testset) // 2, replace=False).tolist() # tolist to obtain Sequence type 
        finetune_set = Subset(testset, finetune_idxs)

        # use the other half to test
        test_idxs = list(set(range(len(testset))) - set(finetune_idxs))
        test_set = Subset(testset, test_idxs)
        
        # save original state_dict
        # deepcopy because state_dict holds referneces to the model's internal state
        orig_state_dict = copy.deepcopy(unsup_model.state_dict())

        # load finetuned dict
        finetuned_state_dict = train_server_model(args, finetune_set, device, transform, unsup_model, epochs=finetune_epochs)
        unsup_model.load_state_dict(finetuned_state_dict)

        # test
        loss, top1, top5 = test_server_model(args, test_set, device, transform, unsup_model)

        # reload original state_dict
        unsup_model.load_state_dict(orig_state_dict)

        return loss, top1, top5
    
    # test using whole testset
    loss, top1, top5 = test_server_model(args, testset, device, transform, unsup_model)
    return loss, top1, top5 
    


def FL_criterion(device):
    return nn.CrossEntropyLoss().to(device)

def SimCLR_criterion(device):
    return nn.CrossEntropyLoss(reduction="mean").to(device)

def SimSiam_criterion(device):
    return nn.CosineSimilarity(dim=-1).to(device)

def SimSiam_loss(device, p1, p2, z1, z2):
    criterion = SimSiam_criterion(device)
    return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

def NCE_loss(device, temperature, feature1, feature2):
    # features = (local batch size * 2, out_dim) shape 
    # feature1, feature2 = torch.tensor_split(features, 2, 0)
    # feature1, 2 = (local batch size, out_dim) shape
    feature1, feature2 = F.normalize(feature1, dim=1), F.normalize(feature2, dim=1)
    batch_size = feature1.shape[0]
    LARGE_NUM = 1e9
    
    # each example in feature1 (or 2) corresponds assigned to label in [0, batch_size) 
    labels = torch.arange(0, batch_size, device=device, dtype=torch.int64)
    masks = torch.eye(batch_size, device=device)
    
    
    logits_aa = torch.matmul(feature1, feature1.T) / temperature #similarity matrix 
    logits_aa = logits_aa - masks * LARGE_NUM
    
    logits_bb = torch.matmul(feature2, feature2.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    
    logits_ab = torch.matmul(feature1, feature2.T) / temperature
    logits_ba = torch.matmul(feature2, feature1.T) / temperature
    

    criterion = SimCLR_criterion(device)

    loss_a = criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
    loss = loss_a + loss_b
    return loss