# Deep HDR Imaging
The Keras Implementation of the [Deep HDR Imaging via A Non-Local Network](https://ieeexplore.ieee.org/document/8989959) - TIP 2020
## Content
- [Deep-HDR-Imaging](#zero-dce-tf)
- [Getting Started](#getting-tarted)
- [Running](#running)
- [References](#references)
- [Citations](#citation)

## Getting Started

- Clone the repository

### Prerequisites

- Tensorflow 2.2.0+
- Tensorflow_addons
- Python 3.6+
- Keras 2.3.0
- PIL
- numpy

```python
pip install -r requirements.txt
```

## Running
### Training 
- Preprocess
    - Download the [training data](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Trainingset.zip) and [testing data](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Testset.zip).

    - Run this file to generate data. (Please remember to change path first)

    ```
    python src/create_dataset.py
    ```

- Train NHDRRNet 
    ```
    python main.py
    ```

- Test ZERO_DCE
    ```
    python test.py
    ```
## Usage
### Training
```
usage: main.py [-h] [--images_path IMAGES_PATH] [--test_path TEST_PATH]
               [--lr LR] [--gpu GPU] [--num_epochs NUM_EPOCHS] 
               [--train_batch_size TRAIN_BATCH_SIZE]
               [--display_ep DISPLAY_EP] [--checkpoint_ep CHECKPOINT_EP]
               [--checkpoints_folder CHECKPOINTS_FOLDER]
               [--load_pretrain LOAD_PRETRAIN] [--pretrain_dir PRETRAIN_DIR]
               [--filter FILTER] [--kernel KERNEL]
               [--encoder_kernel ENCODER_KERNEL]
               [--decoder_kernel DECODER_KERNEL]
               [--triple_pass_filter TRIPLE_PASS_FILTER]
```

```
optional arguments: -h, --help                show this help message and exit
                    --images_path             training path
                    --lr                      LR
                    --gpu                     GPU
                    --num_epochs              NUM of EPOCHS
                    --train_batch_size        training batch size
                    --display_ep              display result every "x" epoch
                    --checkpoint_ep           save weights every "x" epoch
                    --checkpoints_folder      folder to save weight
                    --load_pretrain           load pretrained model
                    --pretrain_dir            pretrained model folder
                    --filter                  default filter
                    --kernel                  default kernel
                    --encoder_kernel          encoder filter size
                    --decoder_kernel          decoder filter size
                    --triple_pass_filter      number of filter in triple pass
```

### Testing
```
usage: test_real.py [-h] [--test_path TEST_PATH] [--gpu GPU]
                    [--weight_test_path WEIGHT_TEST_PATH] [--filter FILTER]
                    [--kernel KERNEL] [--encoder_kernel ENCODER_KERNEL]
                    [--decoder_kernel DECODER_KERNEL]
                    [--triple_pass_filter TRIPLE_PASS_FILTER]
```
```
optional arguments: -h, --help                    show this help message and exit
                    --test_path                   test path
                    --weight_test_path            weight test path
                    --filter                      default filter
                    --kernel                      default kernel
                    --encoder_kernel              encoder filter size
                    --decoder_kernel              decoder filter size
                    --triple_pass_filter          number of filter in triple pass
```

#### Result
![INPUT](Test/PAPER/PeopleStanding/262A2866.png =256x) | [INPUT](Test/PAPER/PeopleStanding/262A2867.png =256x) | [INPUT](Test/PAPER/PeopleStanding/262A2868.png =256x) | ![OUTPUT](Test/PAPER/PeopleStanding/hdr.png =256x) |
|:---:|:---:|:---:|:---:|
| input 1 | input 2 | input 3 | output |

<p float="left">
  <img src="Test/PAPER/PeopleStanding/262A2866.png" width="256" />
  <img src="Test/PAPER/PeopleStanding/262A2867.png" width="256" /> 
  <img src="Test/PAPER/PeopleStanding/262A2868.png" width="256" />
  <img src="Test/PAPER/PeopleStanding/hdr.png" width="256" />
</p>

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/LICENSE) file for details

## References
[1] Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement - CVPR 2020 [link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

[3] Low-light dataset - [link](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view)

## Citation
```
    @misc{guo2020zeroreference,
    title={Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
    author={Chunle Guo and Chongyi Li and Jichang Guo and Chen Change Loy and Junhui Hou and Sam Kwong and Runmin Cong},
    year={2020},
    eprint={2001.06826},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
## Acknowledgments

- This repo is the re-production of the original pytorch [version](https://github.com/Li-Chongyi/Zero-DCE)
- Thank you for helping me to understand more about pains that tensorflow may cause.
- Final words:
    - Any ideas on updating or misunderstanding, please send me an email: <vovantu.hust@gmail.com>
    - If you find this repo helpful, kindly give me a star.

