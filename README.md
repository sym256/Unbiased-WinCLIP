# Unbiased-WinCLIP
This is an unbiased version of original [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)

The implementation of CLIP is based on [open_clip](https://github.com/mlfoundations/open_clip)

## Quick start
Zero-shot anomaly detection 
```sh
bash zero_shot.sh
```
Few-shot anomaly detection 
```sh
bash few_shot.sh
```
  
## Performance evaluation
### Few-shot
#### MVTec AD(https://www.mvtec.com/company/research/datasets/mvtec-ad/) (16-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       99.1 |    66.1 |    69.4 |    95.9 |      100   |    99.4 |   100   |
| bottle     |       94.3 |    60.9 |    64.9 |    85.1 |       99.4 |    98.4 |    99.8 |
| hazelnut   |       98.5 |    58.9 |    61.1 |    93.4 |       98   |    95.6 |    99   |
| leather    |       99.2 |    45.4 |    39.3 |    97.8 |      100   |    99.5 |   100   |
| cable      |       86.9 |    28.7 |    22.8 |    65   |       89.2 |    86.3 |    93.4 |
| capsule    |       96.4 |    32.3 |    24.7 |    89.7 |       83.5 |    92.4 |    96.3 |
| grid       |       94.1 |    28.7 |    19.1 |    82.1 |       99.6 |    99.1 |    99.9 |
| pill       |       92.4 |    36.1 |    28.7 |    89.8 |       89.6 |    93.3 |    98   |
| transistor |       90   |    41.2 |    41.1 |    67.5 |       89.6 |    80.9 |    85.7 |
| metal_nut  |       78.5 |    36.5 |    28.7 |    75.3 |       98.2 |    97.4 |    99.6 |
| screw      |       95.9 |    23.5 |    14.4 |    84.5 |       81.5 |    86.8 |    93.1 |
| toothbrush |       96   |    33.6 |    26.3 |    82.8 |       91.4 |    90.6 |    96.6 |
| zipper     |       97   |    46.5 |    40.8 |    90.5 |       86.4 |    90.3 |    95.8 |
| tile       |       91.7 |    53.5 |    46.2 |    77.5 |      100   |    99.4 |   100   |
| wood       |       94.5 |    56.4 |    59.4 |    84.5 |       99   |    96.8 |    99.7 |
| mean       |       93.6 |    43.2 |    39.1 |    84.1 |       93.7 |    93.7 |    97.1 |

#### VisA (16-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       94.8 |    16.8 |     8   |    90.5 |       96.4 |    91   |    96.9 |
| capsules   |       95.9 |    32.3 |    22.4 |    65.4 |       80.2 |    80.2 |    87.8 |
| cashew     |       96.6 |    32.3 |    22.2 |    90.3 |       95.4 |    91.8 |    97.9 |
| chewinggum |       99   |    57   |    54.6 |    85.9 |       97.7 |    94.8 |    99   |
| fryum      |       94.4 |    32.4 |    26   |    86   |       87.7 |    86.6 |    94.4 |
| macaroni1  |       91.5 |    13   |     4   |    78.9 |       85.6 |    78.8 |    87.7 |
| macaroni2  |       91.6 |     4   |     0.9 |    72.1 |       75.4 |    72.4 |    74.6 |
| pcb1       |       96   |    16.1 |     7.3 |    76.8 |       85.6 |    83.7 |    84.5 |
| pcb2       |       91.5 |     6.6 |     2.9 |    66.4 |       59.6 |    67.9 |    57   |
| pcb3       |       92.9 |    13.4 |     7.6 |    77.6 |       68.9 |    71.2 |    68.9 |
| pcb4       |       95.3 |    22.4 |    15.9 |    82.4 |       85.5 |    79   |    85.6 |
| pipe_fryum |       96.4 |    28   |    18.9 |    94.1 |       88   |    85.6 |    94.2 |
| mean       |       94.7 |    22.9 |    15.9 |    80.5 |       83.8 |    81.9 |    85.7 |


