# Facial expression Recognition

This is a course project of Media and Cognition of Department of EE., Tsinghua University. The main topic is facial expression recognition. Our method is based on [ResNet-50 model](https://github.com/KaimingHe/deep-residual-networks), implementing with [caffe](https://github.com/BVLC/caffe). The model is pretrained on [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and finetuned on several other datasets and our self-captured dataset. The final accuracy is 97.7% (on open datasets, see ``report.pdf```) and 66.7% (on self-captured dataset). More details about datasets, implement details, experiment results or something else can be find in ```report.pdf```. 

## Note

- This project using two versions of caffe: one is [offical version](https://github.com/BVLC/caffe) and the other is the [twtygqyy version](https://github.com/twtygqyy/caffe-augmentation) for further data augmentation. Detailed usage is described below.
- The image pre-processing step include face detection and cropping. In this step we use [MTCNN method](https://github.com/ybch14/MTCNN_face_detection_alignment), implementing with Python.
- The bagging test method includes five models, with means a large GPU memory required, or multiple GPUs.

## File list

### ```preprocess```
    - ```preprocess/model```: MTCNN models, including P-Net, R-Net and O-Net trained models.
    - ```preprocess/main.py```: Preprocess main function, with detect faces in images, crop faces region and resize to a fixed size.
### ```pretrain```
    - ```pretrain/pretrain.sh```: Pre-train script, **using offical version caffe**.
    - ```pretrain/Resnet_50_pretrain.prototxt```: Model definition of modified ResNet-50 model (change the last classification output from 1000 to 8).
    - ```pretrain/Resnet_50_solver.prototxt```: Hyper-parameters definition of pre-training.
    - ```pretrain/resnet_pretrain_model.caffemodel```: Pre-trained model on FER2013 dataset.
### ```finetuning```
    - ```finetuning/models/*```: Fine-tuned models, multiple models used for bagging test.
    - ```bagging_test.py```: Test step main function, bagging five models' votes and recognize facial expression for new images.
    - ```train.sh```: Fine-tuning script, **using twtygqyy version caffe**.
    - ```Resnet_50_finetuning.prototxt```: Fine-tuning model definition, **using twtygqyy version caffe**.
    - ```Resnet_finetuning_solver.prototxt```: Hyper-parameters definition of fine-tuning.
    - ```deploy.prototxt```: Deployment model used in test step. This model works fine with any version of caffe.
- ```report.pdf```: the technology report of this project.

## Usage

### Install caffe

First, if you don't have caffe on your machine, clone the two versions of caffe and install:

```
git clone https://github.com/BVLC/caffe.git
cd caffe/
make all -j
cd ../
git clone https://github.com/twtygqyy/caffe-augmentation.git
cd caffe-augmentation/
make all -j
cd ../
```

If you have problems with caffe installation, you can see [here](http://caffe.berkeleyvision.org/installation.html).

### Clone and change path

Then, clone this project and enter the directory:

```
git clone https://github.com/ybch14/Facial-Expression-Recognition-ResNet.git
cd Facial-Expression-Recognition-ResNet/
```

After cloning, change all the undefined paths to paths that you want:

```
$PROJECTDIR: The directory of this project.
$DATASETDIR: The directory of original datasets.
$TRAINDATADIR: The directory of pre-processed data, whether used for pre-training, finetuning or testing depends on your choice.
$PRETRAINDATADIR: The directory of pre-processed pretrain data.
$FINETUNINGDATADIR: The directory of pre-processed finetune data.
$TESTDATADIR: The directory of pre-processed test data.
```

### Pre-process

After changing all the undefined paths correctly, you need to pre-process your data with ```main.py```:

```
cd preprocess/
python main.py
```

```main.py``` read images in ```$DATASETDIR```, crop the face region and save the cropped image to ```$TRAINDATADIR```. 

### Test on your images

After all these above steps, you can run ```bagging_test.py``` to test your own image. Note: before you run the test, you need to double check the GPU indices in ```bagging_test.py```, because the origin definition needs at least four GPUs with at least 3G memory:

```
cd finetuning/
# adjust GPU indices
# ...
# make sure the test can correctly on your machine
python bagging_test.py
```

### Train your own model 

If you want to train your own model with your datasets, you can switch to ```pretrain``` or ```finetuning``` and execute the ```*.sh``` script. Note: if you are following the above steps, you can execute the scripts directly. If you want to use your own caffe, you need to change the caffe paths:

```
cd pretrain/
# make sure your caffe path is correct
./pretrain.sh
```

or

```
cd finetuning/
# make sure your caffe path is correct
./train.sh
```

If you pre-train your own model, remember to change the model path in ```finetuning/train.sh```. If you fine-tuning your own model, remember to put the model in ```finetuning/models/``` and change the model names in ```finetuning/bagging_test.py```.