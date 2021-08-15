# ArmGAN
Adversarial Representation Mechanism Learning for Network Embedding 

## Overview
Here we provide an implementation of ArmGAN in PyTorch, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files;
- `results/` contains the embedding results;
- `layers.py` contains the implementation of a GCN layer;
- `utils.py` contains the necessary processing function.
- `model.py` contains the implementation of a GAE model, discriminator model and mutual information estimator model.
- `optimizer.py` contains the implementation of the reconstruction loss.

Finally, `armgan.py` puts all of the above together and may be used to execute a full training run on Citeseer.

## Reference
If you make advantage of ArmGAN in your research, please cite the following in your manuscript:


## License
Tianjin University
