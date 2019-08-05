### Unsupervised Deep Functional Maps 
An unsupervised approach to Deep Functional Maps implemented in Tensorflow 1.9.0. You can find our paper explaining more details of the implementation here : https://arxiv.org/pdf/1812.03794.pdf


<figure>
  <img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/Compare_SCAPE_1.png" alt=".." title="Texture Tranfer between two FAUST shapes" />

</figure>

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.6 GPU version and Python 3.6 on Ubuntu 16.04, with CUDA 9.0 and cuDNN 7. There are also some dependencies for a few Python libraries for data like `numpy`, `scipy` etc.

### Shape Matching

To train a DFMnet model to obtain matches between shapes without any ground-truth or geodesic distance matrix (using only a shape's Laplacian eigenvalues and eigenvectors and also Descriptors on shapes):

        python train_DFMnet.py

To obtain matches after training for a given set of shapes:

        python test_DFMnet.py
        
Visualization of functional maps at each training step is possible with tensorboard:

        tensorboard --logdir=./Training/

### Prepare Your Own Data
We also wrote Python code to calculate .mat files containing the eigenvalues and eigenvectors of a shape's Laplace-Beltrami operator, in addition to the transposed eigenvectors with mass matrix correction. To be added is the calculation of descriptors. Simply write in command line:

		python get_LB.py

### Tips for running the code
We notice a few tips that can make the learning experience more comfortable :
* Subsampling helps to gain speed, since less vertices are used
* Some variables must be changed depending on the name of the shapes (tr_reg_ for FAUST for example)
* Matches between test shapes must be specified in `test_DFMnet.py`. We prefered incorparating this in the code rather than create a .txt file.
* If you have an OOM error with tensorflow, reduce the batch_size and/or number of vertices chosen on the shapes and/or use less descriptors.
* Be sure to check your directories and make it similar with the code! Think about the shape names, shape directory, training directory, are the shapes numbered on 3 digits or just two (i.e. Shape1 and not Shape01) etc...

This is what our working directory looks like:

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

### Extra images

We even show great generalization upon triangle meshes that are less regular than the original triangle meshes found in FAUST or SCAPE dataset. We insist on the difference in triangulations with the following two images:

<img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/original_faust_example-1.png" width="800"/>

Above is a visualization of the original FAUST shapes, with 6890 vertices. Below shows a remeshed version with only 5000 vertices, less symmetric and more dense when curbature is higher:

<img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/remesh_faust_example-1.png" width="800"/>

We visualize below matches obtained with FMnet on two shapes of different subjects and in different poses: 

<img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/FMnet.png" width="800"/>

and our Unsupervised Learning approach yields better results:

<img src="https://github.com/JM-data/Unsupervised_FMnet/blob/master/Images/OursAll.png" width="800"/>

