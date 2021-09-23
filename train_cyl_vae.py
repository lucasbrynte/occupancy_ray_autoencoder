import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from models.cylinder_ae import CylVAE 
from torchvision import datasets,transforms

import os

import matplotlib.pyplot as plt


def train_net(config):
    
    prior = config["prior"]
    
    VAE = CylVAE(config["layer_structure"],config["sample_structure"],config["kernel_size"])
    
    #VAE.to('cuda')
    
    training_data = datasets.MNIST(
    root=os.path.join(os.getcwd(),"lib/datasets/data_MNIST"),
    train=True,
    download=False,
    transform=transforms.Compose((transforms.ToTensor(),transforms.Normalize(0, 1)))
    
    )
    
    batch_size=128
    trainloader= torch.utils.data.DataLoader(training_data, batch_size=batch_size)

    opt = torch.optim.Adam(VAE.parameters())
    
    rec_loss = torch.nn.MSELoss()
    
    for l in range(config["nbr_epochs"]):
        for k,batch  in enumerate(trainloader):
            img, _ = batch
            q,gen = VAE(img)
        
            lossKL = kl_divergence(prior,q)
        
            lossrec = rec_loss(img,gen)
        
            loss=lossKL.mean()+lossrec
        
            loss.backward()
        
            opt.step()
        
            opt.zero_grad()
        
        
        
    
    
    
        
    

def main():
    
    config={"layer_structure": [1,32,128,32,8] ,
            "sample_structure":[(28,28),(14,14),(7,7),(5,5),(2,2)] , 
            "kernel_size": 2,
            "prior": MultivariateNormal(torch.zeros(4),torch.eye(4)),
            "nbr_epochs":10
            }
    
    train_net(config)
    
main()

