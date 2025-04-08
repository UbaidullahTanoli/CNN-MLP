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

| Epoch | Phase | Loss   | Accuracy | F1-Score | ROC AUC | PR AUC |
|-------|-------|--------|----------|----------|---------|--------|
| 1     | Train | 0.5545 | 0.5881   | 0.6613   | 0.6101  | 0.7294 |
|       | Val   | 0.1300 | 0.6339   | 0.6887   | 0.6723  | 0.7574 |
| 2     | Train | 0.5325 | 0.6202   | 0.6919   | 0.6437  | 0.7574 |
|       | Val   | 0.1286 | 0.6118   | 0.6393   | 0.6820  | 0.7582 |
| 3     | Train | 0.5296 | 0.6114   | 0.6796   | 0.6532  | 0.7682 |
|       | Val   | 0.1304 | 0.6432   | 0.6987   | 0.6788  | 0.7525 |
| 4     | Train | 0.5208 | 0.6351   | 0.7031   | 0.6661  | 0.7785 |
|       | Val   | 0.1283 | 0.6560   | 0.7191   | 0.6923  | 0.7661 |
| 5     | Train | 0.5181 | 0.6289   | 0.6965   | 0.6715  | 0.7858 |
|       | Val   | 0.1261 | 0.6620   | 0.7158   | 0.7026  | 0.7756 |
| 6     | Train | 0.5183 | 0.6341   | 0.7014   | 0.6727  | 0.7821 |
|       | Val   | 0.1284 | 0.5843   | 0.5778   | 0.6956  | 0.7677 |
| 7     | Train | 0.5168 | 0.6386   | 0.7023   | 0.6762  | 0.7887 |
|       | Val   | 0.1293 | 0.5609   | 0.5198   | 0.7018  | 0.7783 |
| 8     | Train | 0.5136 | 0.6407   | 0.7038   | 0.6820  | 0.7908 |
|       | Val   | 0.1250 | 0.6439   | 0.6787   | 0.7076  | 0.7813 |
| 9     | Train | 0.5135 | 0.6390   | 0.7029   | 0.6792  | 0.7898 |
|       | Val   | 0.1269 | 0.6446   | 0.6856   | 0.6972  | 0.7698 |
| 10    | Train | 0.5062 | 0.6383   | 0.7019   | 0.6924  | 0.8033 |
|       | Val   | 0.1246 | 0.6426   | 0.6658   | 0.7161  | 0.7843 |
| 11    | Train | 0.5080 | 0.6469   | 0.7084   | 0.6933  | 0.7937 |
|       | Val   | 0.1260 | 0.6379   | 0.6731   | 0.7057  | 0.7708 |
| 12    | Train | 0.5139 | 0.6356   | 0.7018   | 0.6803  | 0.7919 |
|       | Val   | 0.1255 | 0.6720   | 0.7253   | 0.7145  | 0.7834 |
| 13    | Train | 0.5101 | 0.6440   | 0.7068   | 0.6869  | 0.7927 |
|       | Val   | 0.1267 | 0.6714   | 0.7336   | 0.7105  | 0.7801 |
| 14    | Train | 0.5105 | 0.6499   | 0.7132   | 0.6868  | 0.7941 |
|       | Val   | 0.1267 | 0.6673   | 0.7256   | 0.7096  | 0.7755 |
| 15    | Train | 0.5077 | 0.6452   | 0.7121   | 0.6888  | 0.7965 |
|       | Val   | 0.1259 | 0.6506   | 0.6893   | 0.7052  | 0.7744 |

### Phase 2: MLP + Last CNN Block (Epochs 16–40)

| Epoch | Phase | Loss   | Accuracy | F1-Score | ROC AUC | PR AUC |
|-------|-------|--------|----------|----------|---------|--------|
| 16    | Train | 0.5039 | 0.6517   | 0.7135   | 0.6995  | 0.8062 |
|       | Val   | 0.1271 | 0.6191   | 0.6234   | 0.7064  | 0.7750 |
| 17    | Train | 0.5107 | 0.6417   | 0.7060   | 0.6839  | 0.7949 |
|       | Val   | 0.1247 | 0.6787   | 0.7315   | 0.7233  | 0.7890 |
| 18    | Train | 0.5027 | 0.6490   | 0.7122   | 0.6980  | 0.8055 |
|       | Val   | 0.1239 | 0.6801   | 0.7256   | 0.7217  | 0.7862 |
| 19    | Train | 0.4962 | 0.6618   | 0.7207   | 0.7124  | 0.8124 |
|       | Val   | 0.1241 | 0.6888   | 0.7446   | 0.7286  | 0.7927 |
| 20    | Train | 0.4971 | 0.6611   | 0.7188   | 0.7129  | 0.8085 |
|       | Val   | 0.1226 | 0.6432   | 0.6675   | 0.7332  | 0.7914 |
| 21    | Train | 0.4984 | 0.6541   | 0.7143   | 0.7070  | 0.8087 |
|       | Val   | 0.1247 | 0.6968   | 0.7604   | 0.7307  | 0.7908 |
| 22    | Train | 0.4955 | 0.6619   | 0.7207   | 0.7131  | 0.8096 |
|       | Val   | 0.1271 | 0.6914   | 0.7666   | 0.7221  | 0.7878 |
| 23    | Train | 0.4929 | 0.6683   | 0.7275   | 0.7183  | 0.8193 |
|       | Val   | 0.1234 | 0.6780   | 0.7287   | 0.7321  | 0.7885 |
| 24    | Train | 0.4904 | 0.6741   | 0.7332   | 0.7229  | 0.8228 |
|       | Val   | 0.1234 | 0.6473   | 0.6658   | 0.7283  | 0.7927 |
| 25    | Train | 0.4794 | 0.6785   | 0.7354   | 0.7390  | 0.8362 |
|       | Val   | 0.1234 | 0.6894   | 0.7448   | 0.7376  | 0.8000 |
| 26    | Train | 0.4833 | 0.6805   | 0.7365   | 0.7338  | 0.8287 |
|       | Val   | 0.1229 | 0.6861   | 0.7367   | 0.7372  | 0.7979 |
| 27    | Train | 0.4829 | 0.6772   | 0.7346   | 0.7346  | 0.8298 |
|       | Val   | 0.1239 | 0.6921   | 0.7556   | 0.7387  | 0.7990 |
| 28    | Train | 0.4782 | 0.6818   | 0.7357   | 0.7405  | 0.8357 |
|       | Val   | 0.1222 | 0.6854   | 0.7368   | 0.7394  | 0.7989 |
| 29    | Train | 0.4848 | 0.6802   | 0.7373   | 0.7314  | 0.8298 |
|       | Val   | 0.1226 | 0.6854   | 0.7323   | 0.7336  | 0.7961 |
| 30    | Train | 0.4867 | 0.6703   | 0.7262   | 0.7267  | 0.8254 |
|       | Val   | 0.1218 | 0.6660   | 0.7063   | 0.7364  | 0.7978 |
| 31    | Train | 0.4800 | 0.6775   | 0.7341   | 0.7361  | 0.8354 |
|       | Val   | 0.1233 | 0.6954   | 0.7499   | 0.7376  | 0.7965 |
| 32    | Train | 0.4770 | 0.6802   | 0.7384   | 0.7404  | 0.8351 |
|       | Val   | 0.1218 | 0.6707   | 0.7140   | 0.7360  | 0.7993 |
| 33    | Train | 0.4840 | 0.6711   | 0.7279   | 0.7292  | 0.8292 |
|       | Val   | 0.1223 | 0.6580   | 0.6927   | 0.7322  | 0.7930 |
| 34    | Train | 0.4766 | 0.6785   | 0.7363   | 0.7411  | 0.8380 |
|       | Val   | 0.1221 | 0.6760   | 0.7212   | 0.7383  | 0.7945 |
| 35    | Train | 0.4810 | 0.6782   | 0.7350   | 0.7363  | 0.8317 |
|       | Val   | 0.1227 | 0.6847   | 0.7358   | 0.7345  | 0.7922 |
| 36    | Train | 0.4778 | 0.6760   | 0.7328   | 0.7384  | 0.8348 |
|       | Val   | 0.1215 | 0.6740   | 0.7144   | 0.7385  | 0.7965 |
| 37    | Train | 0.4784 | 0.6802   | 0.7351   | 0.7415  | 0.8367 |
|       | Val   | 0.1219 | 0.6821   | 0.7237   | 0.7370  | 0.7956 |
| 38    | Train | 0.4758 | 0.6802   | 0.7373   | 0.7440  | 0.8370 |
|       | Val   | 0.1223 | 0.6834   | 0.7255   | 0.7375  | 0.7962 |
| 39    | Train | 0.4784 | 0.6830   | 0.7396   | 0.7425  | 0.8322 |
|       | Val   | 0.1223 | 0.6754   | 0.7125   | 0.7361  | 0.7967 |
| 40    | Train | 0.4705 | 0.6862   | 0.7422   | 0.7499  | 0.8436 |
|       | Val   | 0.1215 | 0.6854   | 0.7293   | 0.7420  | 0.7995 |

## Conclusion

![Training Metrics](MLP_CNN_training_metrics.png)

| Metric    | Best Value | Best Epoch | Last Epoch Value | Last Epoch |
|-----------|-----------|------------|------------------|------------|
| Accuracy  | 0.6968    | 21         | 0.6854           | 40         |
| F1-Score  | 0.7666    | 22         | 0.7293           | 40         |
| ROC AUC   | 0.7420    | 34         | 0.7420           | 40         |
| PR AUC    | 0.7995    | 39         | 0.7995           | 40         |

The CNN+MLP model seems to neither underfit nor overfit despite being trained on just ~7500 images with ~1-2M parameters due to various techniques like the following:

**Notable Features:**

- Transfer Learning: Uses a pre-trained model to leverage knowledge from ImageNet
- Progressive Unfreezing: Starts with frozen backbone layers, then gradually unfreezes for fine-tuning
- Class Imbalance Handling: Uses weighted loss function based on class frequencies
- Comprehensive Metrics: Calculates and reports 5 different performance metrics
- Learning Rate Schedules: Uses ReduceLROnPlateau to reduce learning rates when performance plateaus
- Data Augmentation: Applies various transformations to increase effective dataset size
