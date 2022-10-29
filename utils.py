#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import json
import os

# def exp_history():
#     f = open("./history.json", "r")
#     history = json.load(f)
#     f.close()
#     return history

# def alter_wandb_id_history(task_name, wandb_id):
#     if not os.path.isfile("./history.json"):
#         with open("./history.json", "w") as f:
#             json.dump({task_name: wandb_id}, f)
        
#     else:
#         history = exp_history()

#         with open("./history.json", 'w') as f:
#             history[task_name] = wandb_id
#             json.dump(history, f)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class AverageMeter():
    def __init__(self, name):
        self.name = name
        self.values = []
    
    def update(self, value):
        self.values.append(value)
        
    def get_result(self):
        return sum(self.values)/len(self.values)
    
    def reset(self):
        self.values = []

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])  # 0th always reside at 0th
    for key in w_avg.keys():
        for i in range(1, len(w)):
            if w_avg[key].get_device() != w[i][key].get_device():
                w[i][key] = w[i][key].to(w_avg[key].get_device())
            w_avg[key] += w[i][key]
            
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class CheckpointManager():
    def __init__(self, type):
        self.type = type
        if type == "loss":
            self.best_loss = 1E27 
        
        elif type == "top1":
            self.best_top1 = -1E27 

    def _check_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        
        return False
    
    def _check_top1(self, top1):
        if top1 > self.best_top1:
            self.best_top1 = top1
            return True
        
        return False


    def save(self, loss, top1, model_state_dict, checkpoint_path):
        save_dict = {
            "model_state_dict": model_state_dict, 
            # "optim_state_dict": optim_state_dict, 
            "loss": loss, 
            "top1": top1
        }
        if self.type == "loss" and self._check_loss(loss):
            torch.save(save_dict, checkpoint_path)

        elif self.type == "top1" and self._check_top1(top1):
            torch.save(save_dict, checkpoint_path)

        print(f"model saved at {checkpoint_path}")