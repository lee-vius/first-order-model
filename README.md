# Stylizer Module Embedded

* Stylizer Module includes a generator and a discriminator. For the model definition, refer to `modules/stylizer.py` and `modules/stylizer_discriminator.py`.
* The dataloader created for training stylizer is defined in `my_dataset.py`. 
* The logging functions that output results of trainings are defined in `my_logger.py`
* The demonstration of using stylizer module embedded model is written in `my_demo.ipynb`
* The training of stylizer module is written in `train_stylizer.ipynb`
* The training of original First Order Motion model is written in `run_training.ipynb`

For training a stylizer module, you should create source type and target type dataset respectively. The form of dataset folder is the same as the original First Order Motion model. Remember to define your own configure file. The example configuration is in `config/anim-256.yaml`.

For training, you need a pre-trained FOM model trained on real face videos. Here I will provide the link to a pre-trained model trained on vox-celeb: https://drive.google.com/file/d/1rj3OnE8JN8KNTQhdTZ1lM2AQr-yxgDvx/view?usp=sharing

For dataset, I will provide a sample dataset including Disney cartoon face videos and part of vox celeb dataset: 

* cartoon dataset: https://drive.google.com/drive/folders/1-WJ18xEqZqX_TdW4OtGiW-XeisiMuG4p?usp=sharing
* real face dataset: https://drive.google.com/drive/folders/1rOuM_M5q56RUg-ldoOPiOzrcAQ_d1ifw?usp=sharing

The sample results generated by this project can be viewed in the following: https://drive.google.com/drive/folders/1VtBAEVwdOzBhdCsonRVgE4Dk3fSKaIcB?usp=sharing

The trained stylizer module can be fetched from: https://drive.google.com/file/d/1gpqezL8FuIhwh2OJ0-N0UsKQUBpkGKZZ/view?usp=sharing



**Training platform**

The training is completed on Colab platform. Here I will give the notebook that runs on Colab.

* Run the training process for stylizer module: https://colab.research.google.com/drive/1xo-ZKY0b8dETqW3o6aSlXycHZA_R1yse?usp=sharing
* Run the whole pipeline: https://colab.research.google.com/drive/11HUDxocvLwdh-Xv6455XESeIiThuc3rQ?usp=sharing









**The following is the original First Order Motion Model**

<b>!!! Check out our new [paper](https://arxiv.org/pdf/2104.11280.pdf) and [framework](https://github.com/snap-research/articulated-animation) improved for articulated objects</b>

# First Order Motion Model for Image Animation

This repository contains the source code for the paper [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation) by Aliaksandr Siarohin, [Stéphane Lathuilière](http://stelat.eu), [Sergey Tulyakov](http://stulyakov.com), [Elisa Ricci](http://elisaricci.eu/) and [Nicu Sebe](http://disi.unitn.it/~sebe/). 

## Example animations

The videos on the left show the driving videos. The first row on the right for each dataset shows the source videos. The bottom row contains the animated sequences with motion transferred from the driving video and object taken from the source image. We trained a separate network for each task.

### VoxCeleb Dataset
![Screenshot](sup-mat/vox-teaser.gif)
### Fashion Dataset
![Screenshot](sup-mat/fashion-teaser.gif)
### MGIF Dataset
![Screenshot](sup-mat/mgif-teaser.gif)


### Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

### YAML configs

There are several configuration (```config/dataset_name.yaml```) files one for each `dataset`. See ```config/taichi-256.yaml``` to get description of each parameter.


### Pre-trained checkpoint
Checkpoints can be found under following link: [google-drive](https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) or [yandex-disk](https://yadi.sk/d/lEw8uRm140L_eQ).

### Animation Demo
To run a demo, download checkpoint and run the following command:
```
python demo.py  --config config/dataset_name.yaml --driving_video path/to/driving --source_image path/to/source --checkpoint path/to/checkpoint --relative --adapt_scale
```
The result will be stored in ```result.mp4```.

The driving videos and source images should be cropped before it can be used in our method. To obtain some semi-automatic crop suggestions you can use ```python crop-video.py --inp some_youtube_video.mp4```. It will generate commands for crops using ffmpeg. In order to use the script, face-alligment library is needed:
```
git clone https://github.com/1adrianb/face-alignment
cd face-alignment
pip install -r requirements.txt
python setup.py install
```

### Animation demo with Docker

If you are having trouble getting the demo to work because of library compatibility issues,
and you're running Linux, you might try running it inside a Docker container, which would
give you better control over the execution environment.

Requirements: Docker 19.03+ and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
installed and able to successfully run the `nvidia-docker` usage tests.

We'll first build the container.

```
docker build -t first-order-model .
```

And now that we have the container available locally, we can use it to run the demo.

```
docker run -it --rm --gpus all \
       -v $HOME/first-order-model:/app first-order-model \
       python3 demo.py --config config/vox-256.yaml \
           --driving_video driving.mp4 \
           --source_image source.png  \ 
           --checkpoint vox-cpk.pth.tar \ 
           --result_video result.mp4 \
           --relative --adapt_scale
```

### Colab Demo 
@graphemecluster prepared a gui-demo for the google-colab see: ```demo.ipynb```. To run press ```Open In Colab``` button.

For old demo, see ```old-demo.ipynb```.

### Face-swap
It is possible to modify the method to perform face-swap using supervised segmentation masks.
![Screenshot](sup-mat/face-swap.gif)
For both unsupervised and supervised video editing, such as face-swap, please refer to [Motion Co-Segmentation](https://github.com/AliaksandrSiarohin/motion-cosegmentation).


### Training

To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/dataset_name.yaml --device_ids 0,1,2,3
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.
By default the batch size is tunned to run on 2 or 4 Titan-X gpu (appart from speed it does not make much difference). You can change the batch size in the train_params in corresponding ```.yaml``` file.

### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint
```
You will need to specify the path to the checkpoint,
the ```reconstruction``` subfolder will be created in the checkpoint folder.
The generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
Instructions for computing metrics from the paper can be found: https://github.com/AliaksandrSiarohin/pose-evaluation.

### Image animation

In order to animate videos run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode animate --checkpoint path/to/checkpoint
```
You will need to specify the path to the checkpoint,
the ```animation``` subfolder will be created in the same folder as the checkpoint.
You can find the generated video there and its loss-less version in the ```png``` subfolder.
By default video from test set will be randomly paired, but you can specify the "source,driving" pairs in the corresponding ```.csv``` files. The path to this file should be specified in corresponding ```.yaml``` file in pairs_list setting.

There are 2 different ways of performing animation:
by using **absolute** keypoint locations or by using **relative** keypoint locations.

1) <i>Animation using absolute coordinates:</i> the animation is performed using the absolute postions of the driving video and appearance of the source image.
In this way there are no specific requirements for the driving video and source appearance that is used.
However this usually leads to poor performance since unrelevant details such as shape is transfered.
Check animate parameters in ```taichi-256.yaml``` to enable this mode.

<img src="sup-mat/absolute-demo.gif" width="512"> 

2) <i>Animation using relative coordinates:</i> from the driving video we first estimate the relative movement of each keypoint,
then we add this movement to the absolute position of keypoints in the source image.
This keypoint along with source image is used for animation. This usually leads to better performance, however this requires
that the object in the first frame of the video and in the source image have the same pose

<img src="sup-mat/relative-demo.gif" width="512"> 


### Datasets

1) **Bair**. This dataset can be directly [downloaded](https://yadi.sk/d/Rr-fjn-PdmmqeA).

2) **Mgif**. This dataset can be directly [downloaded](https://yadi.sk/d/5VdqLARizmnj3Q).

3) **Fashion**. Follow the instruction on dataset downloading [from](https://vision.cs.ubc.ca/datasets/fashion/).

4) **Taichi**. Follow the instructions in [data/taichi-loading](data/taichi-loading/README.md) or instructions from https://github.com/AliaksandrSiarohin/video-preprocessing. 

5) **Nemo**. Please follow the [instructions](https://www.uva-nemo.org/) on how to download the dataset. Then the dataset should be preprocessed using scripts from https://github.com/AliaksandrSiarohin/video-preprocessing.

6) **VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.


### Training on your own dataset
1) Resize all the videos to the same size e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
We recommend the later, for each video make a separate folder with all the frames in '.png' format. This format is loss-less, and it has better i/o performance.

2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) Create a config ```config/dataset_name.yaml```, in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. Also adjust the number of epoch in train_params.

#### Additional notes

Citation:

```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}
```
