# **PointNet-based Semantic Segmentation Network**

Welcome to the PointNet Semantic Segmentation Project! This project revolves around implementing and training a PointNet-based model for semantic segmentation on point clouds sourced from the Semantic KITTI dataset. The primary goal is to classify individual points within a point cloud into various semantic categories such as cars, persons, or roads.

**PointNet**

Pointnet was first introduced in the 2017 paper "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" by Charles Qi, Hao Su, Kaichun Mo, and Leonidas Guibas. It was one of the first deep learning architectures able to directly process point clouds for 3D shape analysis and understanding. Prior to Pointnet, most methods converted point clouds to voxel grids or multiple projected views before applying 3D ConvNets. Pointnet operates directly on point sets which are unordered and have varying numbers of points. The authors were motivated to develop a simple and unified architecture to consume raw point clouds since they are the most fundamental representation of 3D geometry.


PointNet is designed to process and analyze 3D point clouds, which represent spatial information from the real world. Unlike traditional 2D image-based approaches, PointNet directly takes point clouds as input, eliminating the need for predefined grid structures or voxelization. The model achieves this by employing a novel symmetric function, which ensures invariance to the permutation of input points.

The key idea behind PointNet is to learn a global feature vector that captures the essential information from the entire point cloud. To accomplish this, PointNet utilizes shared multi-layer perceptrons (MLPs) to process each point independently and then aggregates the information using a max-pooling operation. The resulting global feature vector is then fed into fully connected layers for final classification or segmentation tasks.

**Architecture**<br />
The architecture of PointNet is composed of several components, including shared MLPs, max-pooling layers, and fully connected layers. The shared MLPs operate independently on each point in the input point cloud, extracting local features. The max-pooling layer aggregates these local features into a global feature vector. Finally, the global feature vector is processed by fully connected layers to produce the desired output.

The symmetry of PointNet's architecture ensures that the model remains invariant to changes in the order of the input points, making it suitable for processing unstructured point clouds. Additionally, PointNet is capable of handling variable-sized point clouds, making it a versatile solution for 3D data of different scales and complexities.

**How pointnet is different from U-Net:** Pointnet and U-Net are architecturally very different. U-Net is a convolutional encoder-decoder network for image segmentation originally developed for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path enabling precise localization. Skip connections pass high resolution features from the encoder to decoder. In contrast, Pointnet only contains fully connected layers operating on individual points and global max pooling. It does not have an encoder-decoder structure or convolution operations. While U-Net extracts both local and global context through its contracting and expanding paths, Pointnet relies primarily on point-wise MLPs and global max pooling to summarize point cloud information. An advantage Pointnet has over U-Net is the ability to process unordered point clouds without any structure imposed by voxelization or projection. However, U-Net may learn richer feature representations through its convolutional operations.

Pointnet introduced an effective and simple architecture for point cloud analysis. By using point-wise MLPs and global max pooling, Pointnet can directly process unordered point sets in their raw form. This avoids information loss from conversion to intermediary representations like voxels or multiple views. Pointnet achieved strong performance on 3D classification and segmentation benchmarks while using a simple architecture. Follow-up work has built on Pointnet's ideas to develop more sophisticated deep learning architectures for point clouds. Pointnet motivated further research into permutation invariant and hierarchical networks operating on points. It also highlighted the representational power of point clouds versus voxels or meshes. Pointnet's introduction opened the door to applying deep learning directly on point cloud data, enabling progress in 3D perception for robotics, autonomous vehicles, and other applications.

**Project Highlights:**<br />
**Data Preparation:**<br />
Downloading and preprocessing the Semantic KITTI dataset.<br />
Visualizing point clouds to understand the data structure.

**PointNet Implementation:**<br />
Constructing essential components such as the PointNet Encoder and PointNet Module.<br />
Integration of a T-Net to handle geometric transformations.

**Training Pipeline:**<br />
Setting up data loaders for training, validation, and testing.<br />
Defining hyperparameters, loss functions, and optimizers.<br />
Training the PointNet model and evaluating its performance.



