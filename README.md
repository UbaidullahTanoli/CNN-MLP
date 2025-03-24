# CNN-MLP

This project focuses on the predictive power of a CNN combined with a MLP on Medical (Chest X-ray images) dataset to accurately classify an image as normal or not. This is part of a larger project which combines CNN and LLM to learn from images and the corresponding medical reports of the same patient to make highly accurate binary classification. In this project, only the classification layers (MLP) and the last layer of CNN were trained. This project serves as a baseline for comparing CNN+LLM+MLP.

## Data

Indiana University Chest X-ray dataset was used in the project.
The dataset contains 7,470 pairs of images and reports and measures 14.19 GB. The reports were used as ground truth to compare CNN's predictions.
IU X-ray (Demner-Fushman et al., 2016) can be accessed [here](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

## CNN Model

The architecture of EfficientNet-B1 is used to process and convert the images into vector embeddings. EfficientNet-B1 is a Depp CNN pretrained on Medical dataset (ImageNet) and has ~6.5 Million trainable parameters. EfficientNet‑B1 component remains frozen. I've leveraged pre‑trained knowledge from the CNN model but only updated the last layer during training. It's been proven that EfficientNet architecture gives slightly better accuracy per parameter. Refer to the research paper [here](https://arxiv.org/pdf/1905.11946)

## Multi-Layer Perceptron

Transfer Learning is used to avoid overfitting by freezing the backbone of the CNN. The model uses pre-trained Conv Layers as a fixed feature extractor. Only the MLP, which acts as the binary classification layer is trained in the first 15 epochs, in later ones, the last layer of the CNN was trained as well.

## Parameters

| Parameters | Specification |
| --- | --- |
| Optimizer | Adam |
| Loss Function | Cross Entropy |
| Trainable Parameters in Training Phase 1 | 328,962 |
| Trainable Parameters in Training Phase 2 | 1.33M - 2.33M |
| Batch Normalization | True |
| Image Augmentation | True |
| Droput in 2nd Layer MLP | 0.5 |
| Learning Rate | 0.001 |
| Number of Epochs | 40 |
| Train / Test data | K-Fold Cross Validation with k=1 |

## Results

### Phase 1: MLP-Only (Epochs 1–15)

| Epoch | Phase | Loss   | Accuracy | F1-Score |
|-------|-------|--------|----------|----------|
| 1     | Train | 0.5545 | 0.5881   | 0.6613   |
|       | Val   | 0.1300 | 0.6339   | 0.6887   |
| 2     | Train | 0.5325 | 0.6202   | 0.6919   |
|       | Val   | 0.1286 | 0.6118   | 0.6393   |
| 3     | Train | 0.5296 | 0.6114   | 0.6796   |
|       | Val   | 0.1304 | 0.6432   | 0.6987   |
| 4     | Train | 0.5208 | 0.6351   | 0.7031   |
|       | Val   | 0.1283 | 0.6560   | 0.7191   |
| 5     | Train | 0.5181 | 0.6289   | 0.6965   |
|       | Val   | 0.1261 | 0.6620   | 0.7158   |
| 6     | Train | 0.5183 | 0.6341   | 0.7014   |
|       | Val   | 0.1284 | 0.5843   | 0.5778   |
| 7     | Train | 0.5168 | 0.6386   | 0.7023   |
|       | Val   | 0.1293 | 0.5609   | 0.5198   |
| 8     | Train | 0.5136 | 0.6407   | 0.7038   |
|       | Val   | 0.1250 | 0.6439   | 0.6787   |
| 9     | Train | 0.5135 | 0.6390   | 0.7029   |
|       | Val   | 0.1269 | 0.6446   | 0.6856   |
| 10    | Train | 0.5062 | 0.6383   | 0.7019   |
|       | Val   | 0.1246 | 0.6426   | 0.6658   |
| 11    | Train | 0.5080 | 0.6469   | 0.7084   |
|       | Val   | 0.1260 | 0.6379   | 0.6731   |
| 12    | Train | 0.5139 | 0.6356   | 0.7018   |
|       | Val   | 0.1255 | 0.6720   | 0.7253   |
| 13    | Train | 0.5101 | 0.6440   | 0.7068   |
|       | Val   | 0.1267 | 0.6714   | 0.7336   |
| 14    | Train | 0.5105 | 0.6499   | 0.7132   |
|       | Val   | 0.1267 | 0.6673   | 0.7256   |
| 15    | Train | 0.5077 | 0.6452   | 0.7121   |
|       | Val   | 0.1259 | 0.6506   | 0.6893   |

### Phase 2: MLP + Last CNN Block (Epochs 16–40)

| Epoch | Phase | Loss   | Accuracy | F1-Score |
|-------|-------|--------|----------|----------|
| 16    | Train | 0.5039 | 0.6517   | 0.7135   |
|       | Val   | 0.1271 | 0.6191   | 0.6234   |
| 17    | Train | 0.5107 | 0.6417   | 0.7060   |
|       | Val   | 0.1247 | 0.6787   | 0.7315   |
| 18    | Train | 0.5027 | 0.6490   | 0.7122   |
|       | Val   | 0.1239 | 0.6801   | 0.7256   |
| 19    | Train | 0.4962 | 0.6618   | 0.7207   |
|       | Val   | 0.1241 | 0.6888   | 0.7446   |
| 20    | Train | 0.4971 | 0.6611   | 0.7188   |
|       | Val   | 0.1226 | 0.6432   | 0.6675   |
| 21    | Train | 0.4984 | 0.6541   | 0.7143   |
|       | Val   | 0.1247 | 0.6968   | 0.7604   |
| 22    | Train | 0.4955 | 0.6619   | 0.7207   |
|       | Val   | 0.1271 | 0.6914   | 0.7666   |
| 23    | Train | 0.4929 | 0.6683   | 0.7275   |
|       | Val   | 0.1234 | 0.6780   | 0.7287   |
| 24    | Train | 0.4904 | 0.6741   | 0.7332   |
|       | Val   | 0.1234 | 0.6473   | 0.6658   |
| 25    | Train | 0.4794 | 0.6785   | 0.7354   |
|       | Val   | 0.1234 | 0.6894   | 0.7448   |
| 26    | Train | 0.4833 | 0.6805   | 0.7365   |
|       | Val   | 0.1229 | 0.6861   | 0.7367   |
| 27    | Train | 0.4829 | 0.6772   | 0.7346   |
|       | Val   | 0.1239 | 0.6921   | 0.7556   |
| 28    | Train | 0.4782 | 0.6818   | 0.7357   |
|       | Val   | 0.1222 | 0.6854   | 0.7368   |
| 29    | Train | 0.4848 | 0.6802   | 0.7373   |
|       | Val   | 0.1226 | 0.6854   | 0.7323   |
| 30    | Train | 0.4867 | 0.6703   | 0.7262   |
|       | Val   | 0.1218 | 0.6660   | 0.7063   |
| 31    | Train | 0.4800 | 0.6775   | 0.7341   |
|       | Val   | 0.1233 | 0.6954   | 0.7499   |
| 32    | Train | 0.4770 | 0.6802   | 0.7384   |
|       | Val   | 0.1218 | 0.6707   | 0.7140   |
| 33    | Train | 0.4840 | 0.6711   | 0.7279   |
|       | Val   | 0.1223 | 0.6580   | 0.6927   |
| 34    | Train | 0.4766 | 0.6785   | 0.7363   |
|       | Val   | 0.1221 | 0.6760   | 0.7212   |
| 35    | Train | 0.4810 | 0.6782   | 0.7350   |
|       | Val   | 0.1227 | 0.6847   | 0.7358   |
| 36    | Train | 0.4778 | 0.6760   | 0.7328   |
|       | Val   | 0.1215 | 0.6740   | 0.7144   |
| 37    | Train | 0.4784 | 0.6802   | 0.7351   |
|       | Val   | 0.1219 | 0.6821   | 0.7237   |
| 38    | Train | 0.4758 | 0.6802   | 0.7373   |
|       | Val   | 0.1223 | 0.6834   | 0.7255   |
| 39    | Train | 0.4784 | 0.6830   | 0.7396   |
|       | Val   | 0.1223 | 0.6754   | 0.7125   |
| 40    | Train | 0.4705 | 0.6862   | 0.7422   |
|       | Val   | 0.1215 | 0.6854   | 0.7293   |

## Conclusion

![Training_Metrics]('MLP_CNN_training_metrics.png')

The CNN+MLP model seems to neither underfit nor overfit despite being trained on just ~7500 images with ~1-2M parameters due to various techniques like the following:

**Notable Features:**

- Transfer Learning: Uses a pre-trained model to leverage knowledge from ImageNet
- Progressive Unfreezing: Starts with frozen backbone layers, then gradually unfreezes for fine-tuning
- Class Imbalance Handling: Uses weighted loss function based on class frequencies
- Comprehensive Metrics: Calculates and reports 5 different performance metrics
- Learning Rate Schedules: Uses ReduceLROnPlateau to reduce learning rates when performance plateaus
- Data Augmentation: Applies various transformations to increase effective dataset size
