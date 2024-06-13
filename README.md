# Unbiased-WinCLIP
This is an unbiased version of original [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)

The implementation of CLIP is based on [open_clip](https://github.com/mlfoundations/open_clip)

## Quick start
### Requirements
This repository is implemented and tested on Python 3.8 and PyTorch 2.1.2 To install requirements:
```sh
pip install -r requirements.txt
```

### Data
#### MvTec-AD
Download the dataset from [here.](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

#### VisA
Download the dataset from [here.](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

#### Preprocessing
~


#### Quick start
Few-shot anomaly detection 
```sh
bash few_shot.sh
```
  
## Performance evaluation
### Few-shot
#### MVTec AD (16-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|---------:|------:|------:|------:|------------:|------:|------:|
| carpet     |     99.4 |  67.0 |  68.4 |  97.2 |        99.9 |  99.4 | 100.0 |
| bottle     |     98.0 |  70.9 |  73.5 |  92.4 |        99.9 |  99.2 | 100.0 |
| hazelnut   |     99.0 |  64.8 |  60.7 |  95.0 |        99.4 |  97.2 |  99.6 |
| leather    |     99.4 |  51.2 |  47.2 |  98.4 |        99.6 |  99.5 |  99.8 |
| cable      |     96.1 |  54.7 |  49.3 |  85.6 |        91.5 |  90.0 |  95.2 |
| capsule    |     97.8 |  41.3 |  35.5 |  93.8 |        91.1 |  94.3 |  98.0 |
| grid       |     98.6 |  39.9 |  32.5 |  95.0 |        99.9 |  99.1 | 100.0 |
| pill       |     98.3 |  69.8 |  72.8 |  95.0 |        96.0 |  96.1 |  99.2 |
| transistor |     90.6 |  48.2 |  44.0 |  72.2 |        82.7 |  80.0 |  69.4 |
| metal nut  |     96.3 |  73.1 |  71.2 |  93.2 |        99.5 |  98.4 |  99.9 |
| screw      |     97.7 |  25.7 |  16.4 |  89.8 |        77.6 |  89.1 |  88.4 |
| toothbrush |     99.1 |  63.3 |  59.5 |  90.7 |        93.3 |  93.8 |  97.0 |
| zipper     |     98.3 |  57.9 |  58.1 |  94.8 |        94.3 |  92.8 |  97.1 |
| tile       |     96.2 |  65.2 |  57.8 |  88.2 |        99.2 |  97.6 |  99.7 |
| wood       |     95.1 |  57.6 |  58.6 |  91.7 |        99.8 |  99.2 |  99.9 |
| mean       |     97.3 |  56.7 |  53.7 |  91.5 |        94.9 |  95.0 |  96.2 |

#### VisA (16-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       97.4 |    22.4 |    12.8 |    93.5 |       93.6 |    87.4 |    94.5 |
| capsules   |       96.9 |    31.1 |    19.8 |    77.4 |       82.1 |    81.1 |    90.3 |
| cashew     |       98.9 |    58.1 |    48.3 |    87.4 |       92.5 |    91.3 |    96.3 |
| chewinggum |       99.1 |    59.2 |    58.7 |    85   |       95.7 |    94.9 |    98.4 |
| fryum      |       97.2 |    48.7 |    40.4 |    87.7 |       92.8 |    90.7 |    96.8 |
| macaroni1  |       98.3 |    14.4 |     6.4 |    91.8 |       84.6 |    78.1 |    86.4 |
| macaroni2  |       96   |     9   |     2.4 |    86.5 |       72.4 |    71.1 |    76.7 |
| pcb1       |       99.5 |    66.2 |    68.5 |    89.8 |       84   |    78.7 |    82   |
| pcb2       |       96.7 |    19.7 |     9.3 |    79   |       74.2 |    72   |    74.3 |
| pcb3       |       97   |    38.3 |    27.6 |    88   |       79.3 |    76.3 |    78.4 |
| pcb4       |       97.2 |    32.2 |    23.9 |    87.3 |       91.6 |    86.9 |    87.8 |
| pipe_fryum |       98.9 |    53.9 |    46.8 |    95.1 |       99.3 |    99   |    99.7 |
| mean       |       97.8 |    37.8 |    30.4 |    87.4 |       86.8 |    84   |    88.5 |


