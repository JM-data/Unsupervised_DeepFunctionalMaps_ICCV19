### Unsupervised Deep Functional Maps 
An unsupervised approach to Deep Functional Maps implemented in Tensorflow 1.9.0. The goal was to obtain, and then improve upon, the same state-of-the-art results in the following paper https://arxiv.org/abs/1704.08686 by Litany et al.

We were curious to see how well the neural network would behave in an unsupervised setting, in which no geodistance matrices on any shape was given nor ground-truth functional maps or matches. We change perspective by focusing on learning descriptors to improve the functional maps estimation, rather than calculating a soft-map and use the geodesic distance of the target which are high dimensional matrices, possible causes for slowing down training steps and taking up much more space than a functional map expressed in the spectral domain.

We therefore innovate by considering functional maps not only going from the source shape to the target shape, but also from the target shape to the source shape. This helps introduce new penalties in the training to improve on, like two compositions between the functional maps that should yield identity. More penalties natural to functional maps' instrinsic structures are also added as : commutativity with the Laplacian to preserve isometries, orthogonality to be area-preserving and other properties explained in http://www.lix.polytechnique.fr/~maks/papers/fundescEG17.pdf, published by Dorian Nogneng and Maks Ovsjanikov in 2017.

You can find our paper explaining more details of the implementation here : https://arxiv.org/pdf/1812.03794.pdf


<figure>
  <img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/Compare_SCAPE_1.png" alt=".." title="Texture Tranfer between two FAUST shapes" />
</figure>

We even show great generalization upon triangle meshes that are less regular than the original triangle meshes found in FAUST or SCAPE dataset:

<p float="left">
  <img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/original_faust_example.pdf" width="100" />
  <img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/remesh_faust_example.pdf" width="100" /> 
</p>

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.6 GPU version and Python 3.6 on Ubuntu 16.04, with CUDA 9.0 and cuDNN 7. There are also some dependencies for a few Python libraries for data like `numpy`, `scipy` etc.

### Shape Matching

To train a DFMnet model to obtain matches between shapes without any ground-truth or geodesic distance matrix (using only a shape's Laplacian eigenvalues and eigenvectors and also Descriptors on shapes):

        python train_DFMnet.py

To obtain matches after training for a given set of shapes:

        python test_DFMnet.py
        
Visualization of functional maps at each training step is possible with tensorboard:

        tensorboard --logdir=./Training/

![alt-text-1](https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/FMnet.png "FMnet") ![alt-text-2](https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/OursAll.png "Our unsupervised approach")

### Visualization Tools
We will include Matlab code to visualize matches and calculate their geodesic error soon.

### Prepare Your Own Data
We also wrote Python code to calculate .mat files containing the eigenvalues and eigenvectors of a shape's Laplace-Beltrami operator, in addition to the transposed eigenvectors with mass matrix correction. To be added is the calculation of descriptors. Simply write in command line:

		python get_LB.py

### Tips for running the code
We notice a few tips that can make the learning experience more comfortable :
* Subsampling helps to gain speed, since less vertices are used, but experience shows using more vertices is benificial.
* Some variables must be changed depending on the name of the shapes (tr_reg_ for FAUST for example)
* Matches between test shapes must be specified in `test_DFMnet.py`. We prefered incorparating this in the code rather than create a .txt file.
* 4 parameters are used in front of each penalty, which can be be changed according to the data. For example, if the data is mostly non-isometric, then lowering the weight on the functional map's Orthogonality penalty (E2 in `loss_DFMnet.py`) might yield better results. Likewise, experience shows that when very similar shapes are used in training, a lower penalty on the Commutativity with the Laplacian operator (E3 in `loss_DFMnet.py`) might be advised.
* If RAM permits, increasing the percentage of descriptors randomly chosen at each training step can yield better results. However, we were satisfied by only chosing 20%, or less, first for speed reasons but second because the randomness in this choice reminded us of a pooling step. This could add robustness to our model.
* If you have an OOM error with tensorflow, reduce the batch_size and/or number of vertices chosen on the shapes and/or use less descriptors.
* Be sure to check your directories and make it similar with the code! Think about the shape names, shape directory, training directory, are the shapes numbered on 3 digits or just two (i.e. Shape1 and not Shape01) etc...
* This is what our working directory looks like:

```bash
├── DFMnet.py
├── loss_DFMnet.py
├── test_DFMnet.py
├── train_DFMnet.py
├── Shapes
│   ├── tr_reg_000.mat
│   ├── tr_reg_001.mat
│   ├── ...
│   └── tr_reg_099.mat
└── Training
│   ├── checkpoint
│   ├── graph.pbtxt
│   ├── model.ckpt-5000.data-00000-of-00001
│   ├── model.ckpt-5000.index
│   └── model.ckpt-5000.meta
└── Matches
    ├── tr_reg_080-tr_reg_081.mat
    ├── tr_reg_080-tr_reg_082.mat
    ├── ...
    └── tr_reg_098-tr_reg_099.mat
```
