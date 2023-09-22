# Unsupervised Action Segmentation by Joint Representation Learning and Online Clustering (CVPR 2022)

## Overview
This repository contains the official implementation of our CVPR 2022 paper (https://openaccess.thecvf.com/content/CVPR2022/papers/Kumar_Unsupervised_Action_Segmentation_by_Joint_Representation_Learning_and_Online_Clustering_CVPR_2022_paper.pdf).

If you use the code, please cite our paper:
```
@inproceedings{kumar2022unsupervised,
  title={Unsupervised action segmentation by joint representation learning and online clustering},
  author={Kumar, Sateesh and Haresh, Sanjay and Ahmed, Awais and Konin, Andrey and Zia, M Zeeshan and Tran, Quoc-Huy},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20174--20185},
  year={2022}
}
```

For our recent works, please check out our research page (https://retrocausal.ai/research/).


## Installation
### Create Conda environment 
```
 conda create -n tot
```
### Install packages from requirements file
```
 pip install -r requirements.txt
```
#### Folders
For each dataset create separate folder (specify path --dataset_root) where the inner folders structure is as following:
> features/  
> groundTruth/  
> mapping/  
> models/

During testing will be created several folders which by default stored at --dataset_root, change if necessary 
--output_dir 
> segmentation/  
> likelihood/  
> logs/  


## Datasets

#### 50 Salads
- 50Salads features [link](https://drive.google.com/open?id=17o0WfF970cVnazrRuOWE92-OiYHEXTT3)
- 50Salads ground truth [link](https://drive.google.com/open?id=1mzcN9pz1tKygklQOiWI7iEvcJ1vJfU3R)

#### YouTube Instructions
- YouTube Instructions features [link](https://drive.google.com/open?id=1HyF3_bwWgz1QNgzLvN4J66TJVsQTYFTa) 
- YouTube Instructions ground truth [link](https://drive.google.com/open?id=1ENgdHvwHj2vFwflVXosCkCVP9mfLL5lP)

#### Breakfast
- Breakfast features [link](https://drive.google.com/file/d/1DbYnU2GBb68CxEt2I50QZm17KGYKNR1L)
- Breakfast ground truth [link](https://drive.google.com/file/d/1RO8lrvLy4bVaxZ7C62R0jVQtclXibLXU)

#### Desktop Assembly 
- Desktop Assembly features [link](https://drive.google.com/drive/folders/1t-dUAcY4QMbGt6xHEGriOMgSl5TRBXFM?usp=drive_link)
- Desktop Assembly ground truth [link](https://drive.google.com/drive/folders/1Ql3PwcR24hgjxzCX4XGvcQfVlhekqZu1?usp=drive_link)


## Training

#### 50 Salads
- actions: 'rgb' =='all';
```
python data_utils/FS_utils/fs_train.py
```

#### YouTube Instructions
- actions: 'changing_tire', 'coffee', 'jump_car', 'cpr', 'repot'
    use 'all' to train/test on all actions in series
```
python data_utils/YTI_utils/yti_train.py
```

#### Breakfast
- actions: 'coffee', 'cereals', 'tea', 'milk', 'juice', 'sanwich', 'scrambledegg', 'friedegg', 'salat', 'pancake'
    use 'all' to train/test on all actions in series
```
python data_utils/BF_utils/bf_train.py
```

#### Desktop Assembly
- actions: '2020' =='all';
```
python data_utils/DA_utils/da_train.py
```


## Testing
To Evaluate the model, first set the model path in Test.py file for dataset. 
```
 opt.loaded_model_name = 'model path'
```
#### 50 Salads
- actions: 'rgb' =='all';
```
python data_utils/FS_utils/fs_test.py
```

#### YouTube Instructions
- actions: 'changing_tire', 'coffee', 'jump_car', 'cpr', 'repot'
    use 'all' to train/test on all actions in series
```
python data_utils/YTI_utils/yti_test.py
```

#### Breakfast
- actions: 'coffee', 'cereals', 'tea', 'milk', 'juice', 'sanwich', 'scrambledegg', 'friedegg', 'salat', 'pancake'
    use 'all' to train/test on all actions in series
```
python data_utils/BF_utils/bf_test.py
```

#### Desktop Assembly
- actions: '2020' =='all';
```
python data_utils/DA_utils/da_test.py
```

## Number of Subactions (K)

#### 50 Salads
| Granularity level    | # subactions (K) |
| -------------------- | ---------------- |
|        Eval          |       12         |
|        Mid           |       19         |

#### YouTube Instructions
| Activity class name  | # subactions (K) |
| -------------------- | ---------------- |
|        Changing tire |       11         |
|        Making cofee  |       10         |
|        CPR           |        7         |
|        Jump car      |       12         |
|        Repot plant   |        8         |

#### Breakfast
| Activity class name  | # subactions (K) |
| -------------------- | ---------------- |
|        Coffe         |        7         |
|        Cereals       |        5         |
|        Tea           |        7         |
|        Milk          |        5         |
|        Juice         |        8         |
|        Sandwich      |        9         |
|        Scrambledegg  |       12         |
|        Friedegg      |        9         |
|        Salat         |        8         |
|        Pancake       |       14         |
#### Desktop Assembly
| Version    | # subactions (K) |
| -------------------- | ---------------- |
|        Fixed Order   |       23         |
     
