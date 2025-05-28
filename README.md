# Summer student project - instructions

## Setting-up the code 

You need to run this code from either `lxplus` or `lxplus-gpu` machines (the later has access to gpus, we can see which ones are more efficient)

To connect you just run 

``` 
ssh [yourusername]@lxplus.cern.ch
```

``` 
ssh [yourusername]@lxplus-gpu.cern.ch
```

Then you need to set-up `weaver`. `weaver` is a library made by one of our colleagues at CERN that simplifies the handling of the data when training `pytorch` models. The repository with the code can be found [here](https://github.com/hqucms/weaver-core). Please take a look at the README in there to understand what exactly you need to configure to run your own trainings. You can nevertheless ignore the instructions to set it up, because I’m writing them below

### Setting-up `conda`

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

and follow the instructions. IMPORTANT: when it asks you for the prefix for the installation, please add it in your `/eos/` directory (should be something like `/eos/user/x/yourusername/`, where `x` is the first letter of your username). Otherwise `conda` will go into your home directory, which may not be large enough 


When you install `conda`, it also gives you the option to set-up the automatic activation of conda. I suggest you enable this, so that you can run directly the `conda` commands every time you open a new terminal

### Set-up conda environment and install `pytorch`

```
conda create -n weaver python=3.10 

conda activate weaver
```

Install `pytorch` for `gpu`. These are the instructions for CUDA 12.4. You can check the CUDA version by running `nvidia-smi`. 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Install the `weaver` package

You can just install `weaver` getting the code from the github repository and installing it with pip in edit mode 

```
git clone git@github.com:hqucms/weaver-core.git
cd weaver-core
pip install -e .
```


### Setting everything when starting a new terminal

Once you have followed the instructions above, every time you start a new session in `lxplus` or `lxplus-gpu`, you only need to run 

```
conda activate weaver
```

to get everything configured. 


## The input data 

I have copied the data that you need as input in this path 

```
/eos/cms/store/group/cmst3/user/sesanche/SummerStudent
```

I've separated the dataset in two directories: `train` and `test`. I have also put 5\% of each directory in `train_mini` and `test_mini` for shorter tests. The `train` and `test` directories really contain a lot of data, so training a single epoch will take hours, which is not great for debugging.

Input files are in ROOT format (you can check https://root.cern if you are curious but, in principle, we won't use that software to read the files). The only thing you may want to try is to run 

```
root -l [one of the rootfiles]
```

That will open a root prompt and, within there, you can run 
```
GenObjectTree->Print()
```

That will print the list of variables that you have in the file. There you will see that there are different variables `GenObject_[something]`. All these are the features of each particle in each event: `pt`, `eta` and `phi` are the coordinates of the particle momentum, `mass` and `charge` are its mass and the sign of its charge. Then you have several `isJet`, `isLepton`, `isTau` and  `isNeutrino` flags: these are one-hot flags to indicate which type of particle it is. 


Those files have been generated with two types of Monte Carlo generators that we need to classify. To distinguish between the two, there's another variable labeled `isSignal` that is `1` in one case and `0` in another one. 


## Your first training

To run, there are two important files, included in this repository:  `particle_net.py` and `config.yaml`. `particle_net.py` defines the neural network, in this case, with a particle net architecture. Later on you can try modifying this architecture to optimize it. `config.yaml` defines the input variables, we can also try to play a bit with that later on. 



```
weaver --data-train /eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/*.root \
--data-test /eos/cms/store/group/cmst3/user/sesanche/SummerStudent/test_0p01/*.root \
 --data-config config.yaml \
 --network-config particle_net.py \
 --model-prefix first_training \
 --gpus 0 --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 --log logs/train.log
```

If you don't want to use a gpu, just write `--gpus ""` instead of `--gpus 0`. But I really recommend you run this using the gpu. 

Once this is done, the model will be stored in a file `first_training_best_epoch_state.pt` (it stores one model per epoch, and also keeps the best one separately). You can then evaluate the model using the option `--predict` and the passing the model with `--model-prefix first_training_epoch-0_state.pt`. 

```
weaver --predict \
 --data-train /eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/*.root \
--data-test /eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/*.root \
 --data-config config.yaml \
 --network-config particle_net.py \
 --model-prefix first_training_best_epoch_state.pt \
 --gpus 0 --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 --log logs/train.log \
 --predict-output output.root 
```


This will produce a rootfile that will contain the score given by the neural network, and a label indicating whether each event belongs to one sample or the other. 

You can explore the resulting rootfile and make plots using something like the snippet you have [here](plot_simple.py)

## Things to check during your first week

A bit of ''homework'' for your first week. This is mostly to get you started. Don't feel pressed to finish everything in a week, it may take time to master all this :)   

1) Familiarize yourself with the neural network architecture (see the paper here https://arxiv.org/abs/1902.08570). This introduced in `particle_net.py`, which imports the neural network from (https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py)[here]

2) Run the training above and try to optimize the hyperparameters and the architecture. Here you can be as creative as you want!

3) You can also try one of the other models alternative to `ParticleNet` that are already present in the weaver repository (see [ParticleTransformer here](https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py)). We will also try other architectures in the future, I can give you more material on-the-fly if you want to explore this. 

4) Extend the code snippet I gave you above to implement the reweighting method from the paper we discussed. Write a code that allows you to plot variables comparing the two samples "as they come" and one of the samples reweighted to match the other. To add variables ot the output rootfile, you could add variables as `observers` in the yaml file (see [here](https://github.com/jet-universe/particle_transformer/blob/29ef32b5020c11d0d22fba01f37a740a72cbbb4d/data/JetClass/JetClass_full.yaml#L83-L94)). Some interesting variables you could plot: number of particles that are jets in each event, the `pt=np.sqrt(px**2+py**2)` of the leptons in the event, the `pt` of the jets in the event. 
