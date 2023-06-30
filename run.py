import subprocess
import numpy as np



seeds = np.arange(10)
for seed in seeds:
    for samples in [500]:
        for epochs in [200]:
            for noise in [0.2]:
            
                dim = 10
                subprocess.call('python single-neuron-methods.py --result-dir results-plot-correlated-noise --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer adam', shell=True)
                subprocess.call('python single-neuron-methods.py --result-dir results-plot-correlated-noise --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer sgd', shell=True)
                subprocess.call('python single-neuron-methods.py --result-dir results-plot-correlated-noise --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer lbfgs', shell=True)
                subprocess.call('python single-neuron-methods.py --result-dir results-plot-correlated-noise --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer newton', shell=True)
            

# for seed in seeds:
#     for samples in [500]:
#         for epochs in [5, 10, 100, 200, 500]:
#             for noise in [0.2]:
            
#                 dim = 10
#                 subprocess.call('python single-neuron-methods.py --result-dir results-epochs-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer adam', shell=True)
#                 subprocess.call('python single-neuron-methods.py --result-dir results-epochs-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer sgd', shell=True)
#                 subprocess.call('python single-neuron-methods.py --result-dir results-epochs-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer lbfgs', shell=True)
#                 subprocess.call('python single-neuron-methods.py --result-dir results-epochs-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer newton', shell=True)

# for seed in seeds:
#     for dim in [1, 2, 5, 10, 50]:
#         for epochs in [100]:
#             for noise in [0.2]:
#                 samples = 500
#                 subprocess.call('python single-neuron-methods.py --result-dir results-dim-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer adam', shell=True)
#                 subprocess.call('python single-neuron-methods.py --result-dir results-dim-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer sgd', shell=True)
#                 subprocess.call('python single-neuron-methods.py --result-dir results-dim-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer lbfgs', shell=True)
#                 subprocess.call('python single-neuron-methods.py --result-dir results-dim-ablate --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --optimizer newton', shell=True)
            