import numpy as np 
import torch
from torch import nn 
import torchvision
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torchmin import Minimizer

import random
import copy
import argparse
import sys
import yaml
import time
import pathlib

# args = None
def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch single neuron case")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument(
        "--epochs",
        default=100000,
        type=int,
        help="number of total epochs to run")

    parser.add_argument("--num-samples", default=100, type=int, help='number of samples')
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="initial learning rate"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    
    parser.add_argument("--dim", type=int, default=10, help="input dim / overparam")
    parser.add_argument("--noise", type=float, default=0.001, help="noise")
    parser.add_argument("--name", type=str, default='trial', help="name of the experiments")
    parser.add_argument("--result-dir", type=str, default='results-finetune-theory', help="result dir of the experiments")

    args = parser.parse_args()

    return args

args = parse_arguments()
print(args)
################ Save Results to CSV
def write_result_to_csv(**kwargs):
    filename = args.result_dir + '.csv'

    results = pathlib.Path(filename)
    
    if not results.exists():
        with open(results, 'w', newline=''):

            results.write_text(
                "Name, "
                "Dim,"
                "Noise,"
                "Epochs,"
                "Optimizer,"
                "Train Loss,"
                "Test Loss,"
                "Num Samples,"
                "Weight Init,"
                "Weight Learnt,"
                "A init,"
                "A learnt\n"
            )


    with open(results, "a+") as f:
        f.write(
            (
                "{name}, "
                "{dim}, "
                "{noise:.05f}, "
                "{epochs}, "
                "{optimizer}, "
                "{train_loss:.05f}, "
                "{test_loss:.05f}, "
                "{num_samples}, "
                "{weight_init:.05f}, "
                "{weight_learnt:.05f},"
                "{a_init:.05f},"
                "{a_learnt:.05f}\n"
            ).format(**kwargs)
        )


##############

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Generate data
#number of samples
n=args.num_samples
#input dim
d=args.dim
#input data drawn from U[-1,1]^d
x = 2*torch.rand([n,d],dtype=float)-1
#additive noise
sigma= args.noise #0.5 #50 #0.5 #0 #0.5
# The true y label is the first dimension input multiplied by the relu
# y = F.relu(x[:,0]) + torch.randn(n,dtype=float)*sigma
# make the noise correlated with the data
y = F.relu(x[:,0]) + torch.randn(n,dtype=float)*x[:,0]

###### Creating Test data
ntest = int(0.2 * n)
xtest = 2*torch.rand([ntest,d],dtype=float)-1
# ytest = F.relu(xtest[:,0]) + torch.randn(ntest,dtype=float)*sigma
ytest = F.relu(xtest[:,0]) + torch.randn(ntest,dtype=float)*xtest[:,0]


###### Define Model #############
    
class neuron(nn.Module):
    def __init__(self, in_dim, out_dim, width):
        super(neuron, self).__init__()
        self.layer1 = nn.Linear(in_dim, width, bias= False)
        self.layer2 = nn.Linear(width, out_dim, bias = False)
        print(self.layer2.weight.shape)
        self.initialize()
        print(self.layer2.weight.shape)
    def forward(self, x):
        # forward pass
        x = F.relu(self.layer1(x))
        
        x = self.layer2(x)
        
        return x
    
    def initialize(self):
        in_dim, width = self.layer1.weight.shape
        _, out_dim = self.layer2.weight.shape
        
        self.layer1.weight.data = torch.randn((in_dim, width),dtype=float)/np.sqrt(width)
        
        # w[0,0]=-1
        self.layer2.weight.data[0] = (torch.rand(1)[0]*2-1) * torch.sqrt(torch.sum(self.layer1.weight.data**2))
        


model = neuron(d, 1, 1)
model.to(torch.double)
print('The model is defined as')
print(model)
# save the weight of the true weight at init
weight_init = model.layer1.weight.data[0,0].clone()
a_init = model.layer2.weight.data[0].clone().item()
#################
# Defining Magnitude Pruner



########Train the Model

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.0) #torch.optim.Adam(model.parameters(), lr=0.00001)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
if args.optimizer == 'lbfgs':
    optimizer = torch.optim.LBFGS(model.parameters())
if args.optimizer == 'newton':
    optimizer = Minimizer(model.parameters(), method='newton-exact', tol=1e-6, max_iter=1000, disp=2)

loss_criterion = nn.MSELoss()

#before training
print('weights before training')
print(model.layer1.weight)
print(model.layer2.weight)
output = model(x)
torch.save(model.state_dict(),"model_{}_init.pt".format(args.name))
torch.save(optimizer.state_dict(),"optimizer_{}_init.pt".format(args.name))

output=model(x)
outMin = output
minval = 1



######## Training to Convergence

train_loss_list = []
test_loss_list = []

if args.optimizer == 'sgd' or args.optimizer == 'adam':
    best_train_loss = 1
    best_test_loss = 1

    print('Final Training Begins with first order method')
    for epoch in range(args.epochs):
        output = model(x)
        loss = loss_criterion(output.flatten(), y.flatten())
        train_loss_list.append(loss)
        if torch.mean(loss) < best_train_loss:
            best_train_loss = torch.mean(loss)
            outMin = output
        loss.backward()

        optimizer.step()

        if epoch%100==0:
            print("Epoch " + str(epoch) + ": " + str(torch.mean(loss).item()) + " min: " + str(minval))
        
        print('evaluating the model')
        output_test = model(xtest)
        test_loss = loss_criterion(output_test.flatten(), ytest.flatten())
        test_loss_list.append(test_loss)
        if torch.mean(test_loss) < best_test_loss:
            best_test_loss = torch.mean(test_loss)

if args.optimizer == 'lbfgs':
    best_train_loss = 1
    best_test_loss = 1

    print('Training begins with a LBFGS')
    for epoch in range(args.epochs):
        output = model(x)
        loss = loss_criterion(output.flatten(), y.flatten())
        train_loss_list.append(loss)
        if torch.mean(loss) < best_train_loss:
            best_train_loss = torch.mean(loss)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = loss_criterion(output, y)
            loss.backward()
            return loss

        optimizer.step(closure)

        print('evaluating the model')
        output_test = model(xtest)
        test_loss = loss_criterion(output_test.flatten(), ytest.flatten())
        test_loss_list.append(test_loss)
        if torch.mean(test_loss) < best_test_loss:
            best_test_loss = torch.mean(test_loss)

if args.optimizer == 'newton':

    best_train_loss = 1
    best_test_loss = 1

    print('Training begins with a Newton')
    for epoch in range(args.epochs):

        output = model(x)
        loss = loss_criterion(output.flatten(), y.flatten())
        train_loss_list.append(loss)
        if torch.mean(loss) < best_train_loss:
            best_train_loss = torch.mean(loss)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = loss_criterion(output, y)
            return loss

        optimizer.step(closure)

        print('evaluating the model')
        output_test = model(xtest)
        test_loss = loss_criterion(output_test.flatten(), ytest.flatten())
        test_loss_list.append(test_loss)

        if torch.mean(test_loss) < best_test_loss:
            best_test_loss = torch.mean(test_loss)

output = model(x)
print("train loss:")
print(loss_criterion(output.flatten(), y.flatten()))
print("test loss:")
output_test = model(xtest)
print(loss_criterion(output_test.flatten(), ytest.flatten()))
print("================")
#print(model)
print('Printing Model Weights after training')
print(model.layer1.weight)
print(model.layer2.weight)
weight_learnt = model.layer1.weight.data[0,0].clone()
a_learnt = model.layer2.weight.data[0].clone().item()

print('Weight 1 init and learnt: ', weight_init, weight_learnt)

torch.save(train_loss_list,"train_loss_correlated_noise_{}_{}_{}_{}_{}.pt".format(args.optimizer, args.epochs, args.noise, args.dim, args.seed))
torch.save(test_loss_list,"test_loss_correlated_noise_{}_{}_{}_{}_{}.pt".format(args.optimizer, args.epochs, args.noise, args.dim, args.seed))

write_result_to_csv(
        name=args.name,
        dim=args.dim,
        noise=args.noise,
        epochs=args.epochs,
        optimizer=args.optimizer,
        train_loss=best_train_loss,
        test_loss=best_test_loss,
        num_samples = args.num_samples,
        weight_init = weight_init,
        weight_learnt = weight_learnt,
        a_init = a_init,
        a_learnt = a_learnt
    )