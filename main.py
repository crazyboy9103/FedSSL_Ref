#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
from multiprocessing import Manager
import torch.multiprocessing as mp
import wandb
from datetime import datetime

from options import args_parser
from trainers import test_server_model, test_client_model, train_client_model, train_server_model
from models import ResNet_model
from utils import average_weights, CheckpointManager, set_seed
from data_utils import SimCLRTransformWrapper, SimCLRTransform, NoTransform, get_dataset, get_loader, CustomData


mp.set_start_method('spawn', force=True)
if __name__ == '__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

    args = args_parser()
    
    # task_name = f"{args.exp}_iid_{args.agg}" if args.iid == True else f"{args.exp}_noniid_{args.agg}"
    

    # alter_wandb_id_history(task_name, wandb_name)
    wandb_name = now if args.wandb_tag == "" else args.wandb_tag
    wandb_writer = wandb.init(
        name = wandb_name,
        project = "Fed", 
        resume = "auto",
        id = now,
        # mode = "offline"
        # mode="disabled"|"offline"
    )

    
    set_seed(args.seed)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Set the model to train and send it to device.
    server_model   = ResNet_model(args).to(device)
    client_model = ResNet_model(args).to(device)
    
    # Get datasets 
    # train_custom_sets: {client_id: CustomData}
    # server_data_idx: {class: list(idxs)}
    # train_dataset: torchvision.datasets.CIFAR10
    # test_dataset:  torchvision.datasets.CIFAR10
    train_custom_sets, server_data_idx, train_dataset, test_dataset = get_dataset(args)

    # number of participating clients
    num_clients_part = int(args.frac * args.num_users)
    assert num_clients_part > 0
    
    # Save checkpoint of best model based on (top1 / loss) so far
    ckpt_manager = CheckpointManager(args.ckpt_criterion)

    # train_server_model(args, trainset, device, transform, sup_model, epochs=1)
    # train_client_model(args, trainset, device, transform, sup_model = None, unsup_model = None, epochs=1, q=None)
    # test_server_model(args, testset, device, transform, sup_model)
    # test_client_model(args, testset, device, transform, unsup_model, finetune=True, finetune_epochs = 5)

    # Training 
    for epoch in range(args.epochs):
        if args.exp != "centralized":        
            ### 
            # args.agg
            # lower : Use sup data at server to train, then apply contrastive learning at clients on the same model
            # FedSSL : Same as lower, but MSE regularization between supervised model and unsupervised model at client 
            # upper : Use sup data at server to train, then apply pseudo-label, fixmatch, BYOL, UDA ... to train
            
            iid_server_data_idx = []
            for _class, idxs in server_data_idx.items():
                rand_idxs = np.random.choice(idxs, args.server_num_items, replace=False)
                iid_server_data_idx.extend(rand_idxs.tolist())

            server_set = CustomData(args, iid_server_data_idx, train_dataset)

            state_dict = train_server_model(
                args = args,
                trainset = server_set,
                device = device,
                transform = NoTransform(args),
                sup_model = server_model,
                epochs = args.server_epochs
            )

            server_model.load_state_dict(state_dict)
            
            loss, top1, top5 = test_server_model(
                args = args, 
                testset = test_dataset,
                device = device,
                transform = NoTransform(args), 
                sup_model = server_model
            )


            wandb_writer.log({
                "epoch": epoch,
                "sup_test_loss_server": loss, 
                "sup_top1_server": top1, 
                "sup_top5_server": top5
            })

            print(f'\n | Server Trained : test_loss {loss} top1 {top1}% top5 {top5}% |\n')         
            # --------------------------------------------------------------------------------------------
            # Client train
            local_weights = {}
            
            # Select clients for training in this round
            part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
            
            # multiprocessing queue
            processes = []
            
            done = mp.Event()
            q = mp.Queue()
            for i, client_id in enumerate(part_users_ids):
                trainset = train_custom_sets[client_id]
                curr_device = torch.device(f"cuda:{i % torch.cuda.device_count()}")

                if args.parallel:
                    p = mp.Process(target = train_client_model, args=(
                        args, 
                        trainset, 
                        curr_device, 
                        SimCLRTransformWrapper(args) if args.exp != "FLSL" else NoTransform(args),
                        server_model,
                        client_model,
                        args.local_ep,
                        q,
                        done, # requires done event to wait until tensor is accessed from parent process
                              # https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
                    ))
                    p.start()
                    processes.append(p)

                else:
                    client_state_dict = train_client_model(
                        args = args, 
                        trainset = trainset, 
                        device = curr_device, 
                        transform = SimCLRTransformWrapper(args) if args.exp != "FLSL" else NoTransform(args), 
                        sup_model = server_model, 
                        unsup_model = client_model, 
                        epochs = args.local_ep, 
                        q = None
                    )

                    local_weights[i] = client_state_dict

            if args.parallel:
                # q: mp.Queue is blocking object, thus works as join
                for i in range(num_clients_part):
                    weight = q.get()        
                    local_weights[i] = weight
                    del weight
                done.set()

            unsup_weights = average_weights(local_weights)
            client_model.load_state_dict(unsup_weights)

            loss, top1, top5 = test_client_model(
                args = args,
                testset = test_dataset,
                device = device,
                transform = NoTransform(args), 
                unsup_model = client_model,
                finetune = args.finetune, 
                finetune_epochs = args.finetune_epoch
            )

        # if centralized 
        else:
            iid_server_data_idx = []
            for _class, idxs in server_data_idx.items():
                rand_idxs = np.random.choice(idxs, args.server_num_items, replace=False)
                iid_server_data_idx.extend(rand_idxs.tolist())

            server_set = CustomData(args, iid_server_data_idx, train_dataset)

            state_dict = train_server_model(
                args = args,
                trainset = server_set,
                device = device,
                transform = NoTransform(args),
                sup_model = server_model,
                epochs = args.server_epochs
            )

            sup_model.load_state_dict(state_dict)
            
            loss, top1, top5 = test_server_model(
                args = args, 
                testset = test_dataset,
                device = device,
                transform = NoTransform(args), 
                sup_model = server_model
            )
            
        print("#######################################################")
        print(f' \nAvg Validation Stats after {epoch+1} global rounds')
        print(f'Validation Loss     : {loss:.2f}')
        print(f'Validation Accuracy : top1/top5 {top1:.2f}%/{top5:.2f}%\n')
        print("#######################################################")

        wandb_writer.log({
            "test_loss_server": loss, 
            "top1_server": top1,
            "top5_server": top5,
            "epoch": epoch
        })
