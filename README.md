# WaSR-T: Temporal Context for Robust Maritime Obstacle Detection
[[`arxiv`](https://arxiv.org/abs/2203.05352)] [[`BibTeX`](#cite)] [[`weights`](#weights)] [[`data`](#data)]

This is the official PyTorch implementation of the [WaSR-T network](https://arxiv.org/abs/2203.05352) [[1](#ref-wasrt)]. Contains scripts for training and running the network and weights pretrained on the MaSTr1325 [[2](#ref-mastr)] (and MaSTr1478) dataset. 

Our work will be presented at the *IROS 2022* conference in Kyoto, Japan.

<p align="center">
    <img src="figures/comparison.gif" alt="Comparison WaSR - WaSR-T">
    Comparison between WaSR (single-frame) and WaSR-T (temporal context) on hard examples.
</p>


## About WaSR-T

WaSR-T is a temporal extension of the established [WaSR model](https://github.com/lojzezust/WaSR) [[3](#ref-wasr)] for maritime obstacle detection. It harnesses the temporal context of recent image frames to reduce the ambiguity on reflections and improve the overall robustness of predictions.

<p align="center">
    <img src="figures/architecture.png" alt="WaSR-T architecture">
</p>

The target and context (i.e. previous) frames are encoded using a shared encoder network. To extract the temporal context from the past frames, we apply a 3D convolution operation over the temporal dimension. The 3D convolution is able to extract discriminative information about local texture changes over the recent frames. The resulting temporal context is concatenated with the target frame features and passed to the decoder, which produces the final predictions.

## Setup

**Requirements**: Python >= 3.6 (tested on Python 3.8), PyTorch 1.8.1, PyTorch Lightning 1.4.4 (for training)

The required Python libraries can be installed using the following pip command

```bash
pip install -r requirements.txt
```

## Usage

The WaSR-T model used in our experiments (ResNet-101 backbone) can be initialized with the following code.
```python
from wasr_t.wasr_t import wasr_temporal_resnet101

model = wasr_temporal_resnet101(num_classes=3)
```

WaSR-T model operates in two different modes: 
- **sequential**: An online mode, useful for inference. Only one frame is processed at a time. Features of the previous frames are stored in a circular buffer. The frames must be processed one after the other, thus the batch size must be 1. The context buffer is initialized from copies of the first frame in the sequence.
- **unrolled**: An offline mode, used during training. Each sample consists of a target frame and the required previous frames. Supports batched processing.

You can switch between the two modes by calling `sequential()` or `unrolled()` on the model.

Example of sequential operation:
```python
model = model.sequential()

model.clear_state() # Clear the temporal buffer of the model
for image in sequence:
    # image is a [1,3,H,W] tensor
    output = model({'image': image})
```
**Note**: If you run inference on multiple sequences you must call `clear_state()` on the model to clear the buffer before moving to a new sequence. Otherwise the context of the last frames of the previous sequence will be used, which may lead to faulty predictions.

Example of unrolled operation:
```python
model = model.unrolled()

# images is a batch of images: [B,3,H,W] tensor, where B is the batch size
# hist_images is a batch of context images: [B,T,3,H,W], where T is the number of context frames used by the network (default 5)
output = model({'image': images, 'hist_images': hist_images})
```

## Model inference

To run sequential WaSR-T inference on a sequence of image frames use the `predict_sequential.py` script.

```bash
# export CUDA_VISIBLE_DEVICES=-1 # CPU only
export CUDA_VISIBLE_DEVICES=0 # GPU to use
python predict_sequential.py \
--sequence-dir examples/sequence \
--weights path/to/model/weights.pth \
--output-dir output/predictions
```

The script will loop over the images in the `--sequence-dir` directory in alphabetical order. Predictions will be stored as color-coded masks to the specified output directory.

If you wish to run inference on a video file, first convert the file to a sequence of images. For example, using *ffmpeg*:
```bash
mkdir sequence_images
ffmpeg -i video.mp4 sequence_images/frame_%05d.jpg
```


## <a name="weights"></a>Model weights

Currently available pretrained model weights. All models are evaluated on the MODS benchmark [[4](#ref-mods)]. F1 scores overall and inside the danger zone are reported in the table.

| backbone   | T | dataset   | F1       | F1<sub>D</sub> | weights                                                                            |
|------------|---|-----------|----------|----------|------------------------------------------------------------------------------------|
| ResNet-101 | 5 | MaSTr1325 | 93.7     | 87.3     | [link](https://github.com/lojzezust/WaSR-T/releases/download/weights/wasrt_mastr1325.pth) |
| ResNet-101 | 5 | MaSTr1478 | **94.4** | **93.6** | [link](https://github.com/lojzezust/WaSR-T/releases/download/weights/wasrt_mastr1478.pth) |


## Model training

To train your own models, use the `train.py` script. For example, to reproduce the results of our experiments use the following steps:

1. Download and prepare the [MaSTr1325 dataset](https://box.vicos.si/borja/viamaro/index.html#mastr1325) (images and GT masks). Also download the context frames for the MaSTr1325 images [here](#data).
2. Edit the dataset configuration files (`configs/mastr_1325_train.yaml`, `configs/mastr1325_extra` and `configs/mastr1325_val.yaml`) so that they correctly point to the dataset directories.
3. Use the `train.py` to train the network.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # GPUs to use
python train.py \
--train_config configs/mastr1325_train.yaml \
--val_config configs/mastr1325_val.yaml \
--validation \
--model_name my_wasr \
--batch_size 2 \
--epochs 100
```

**Note:** Model training requires a large amount of GPU memory (>11 GB per GPU). If you use smaller GPUs, you can reduce the memory consumption by decreasing the number of backbone backpropagation steps (`--backbone-grad-steps`) or using a smaller context length (`--hist-len`).

### Logging and model weights

A log dir with the specified model name will be created inside the `output` directory. Model checkpoints and training logs will be stored here. At the end of the training the model weights are also exported to a `weights.pth` file inside this directory.

Logged metrics (loss, validation accuracy, validation IoU) can be inspected using tensorboard.

```bash
tensorboard --logdir output/logs/model_name
```

## <a name="data"></a>Data

We extend the MaSTr1325 dataset by providing the context frames (5 preceding frames). We also extend the dataset with additional hard examples to form MaSTr1478.
- MaSTr1325 context frames: [link](https://github.com/lojzezust/WaSR-T/releases/download/weights/mastr1325_context.zip)
- MaSTr1478 extension data: [link](https://github.com/lojzezust/WaSR-T/releases/download/weights/mastr153.zip)

## <a name="cite"></a>Citation

If you use this code, please cite our paper:

```bib
@InProceedings{Zust2022Temporal,
  title={Temporal Context for Robust Maritime Obstacle Detection},
  author={{\v{Z}}ust, Lojze and Kristan, Matej},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022}
}
```

## References

<a name="ref-wasrt"></a>[1] Žust, L., & Kristan, M. (2022). Temporal Context for Robust Maritime Obstacle Detection. 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

<a name="ref-mastr"></a>[2] Bovcon, B., Muhovič, J., Perš, J., & Kristan, M. (2019). The MaSTr1325 dataset for training deep USV obstacle detection models. 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

<a name="ref-wasr"></a>[3] Bovcon, B., & Kristan, M. (2021). WaSR--A Water Segmentation and Refinement Maritime Obstacle Detection Network. IEEE Transactions on Cybernetics

<a name="ref-mods"></a>[4] Bovcon, B., Muhovič, J., Vranac, D., Mozetič, D., Perš, J., & Kristan, M. (2021). MODS -- A USV-oriented object detection and obstacle segmentation benchmark.
