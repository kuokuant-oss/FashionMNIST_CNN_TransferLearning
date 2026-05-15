# Fashion MNIST 圖像分類

使用 PyTorch 訓練多種神經網路架構，對 Fashion MNIST 服裝圖像資料集進行分類。

## 資料集

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) 包含 70,000 張 28×28 灰階服裝圖片，分為 10 類：

| Label | 類別 |
|-------|------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

- 訓練集：60,000 張
- 測試集：10,000 張

## 環境需求

```
python >= 3.8
torch
torchvision
matplotlib
numpy
scikit-learn
seaborn
datasets (HuggingFace)
medmnist
```

安裝依賴：

```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn datasets medmnist
```

## 作業內容

### Question 1 — 資料視覺化
建立 `train_loader`、`val_loader`、`test_loader`（batch size = 64），並視覺化樣本圖片。

### Question 2 — 簡單神經網路 (SimpleNN)
單一隱藏層（64 個節點）的全連接網路。

| 子題 | 內容 |
|------|------|
| 2a | 實作 `SimpleNN` 類別 |
| 2b | 實作 `train_one_epoch` 函式 |
| 2c | 實作 `evaluate` 函式（回傳 loss 與 accuracy） |
| 2d | 實作 `train` 函式（Adam + CrossEntropyLoss，含 validation） |
| 2e | 訓練 ≥10 個 epoch，儲存至 `simple_nn.pt`，繪製準確率曲線 |
| 2f | 載入模型，計算測試集準確率（目標 > 82%） |

### Question 3 — 雙層感知器 (TwoLayerNN)
兩層全連接隱藏層，每層後接 Dropout。

| 子題 | 內容 |
|------|------|
| 3a | 實作 `TwoLayerNN`（hidden_sizes=(512,256), dropout=0.2） |
| 3b | 訓練 ≥10 個 epoch，儲存至 `two_layer_nn.pt`（目標訓練準確率 > 89%） |
| 3c | 載入模型，計算測試集準確率（目標 > 87%） |

### Question 4 — 卷積神經網路 (CNN)
三層卷積層（含 BatchNorm + ReLU + MaxPooling），接三層全連接層（前兩層含 Dropout）。

| 子題 | 內容 |
|------|------|
| 4a | 實作 `CNN` 類別 |
| 4b | 訓練 ≥10 個 epoch，儲存至 `cnn.pt`（目標訓練準確率 > 93%） |
| 4c | 載入模型，計算測試集準確率（目標 > 90%） |

### Question 5 — 模型討論
比較三種模型的表現，分析訓練曲線，討論過擬合/欠擬合與邊際報酬遞減。

## 延伸分析（自選）

- **混淆矩陣**：視覺化各類別的預測錯誤分布
- **錯誤樣本分析**：展示被誤判的圖片
- **遷移學習**：將 Fashion MNIST 預訓練的 CNN 遷移至 KMNIST（平假名）與 PneumoniaMNIST（胸部 X 光）
  - 比較 Frozen vs. Unfrozen 兩種策略
  - 含 Early Stopping（patience=5）
- **低資料量實驗**：以 10% KMNIST 訓練資料評估遷移學習效益

## 儲存的模型檔案

| 檔案 | 說明 |
|------|------|
| `simple_nn.pt` | 單層全連接網路權重 |
| `two_layer_nn.pt` | 雙層全連接網路權重 |
| `cnn.pt` | 卷積神經網路權重 |

## 執行方式

以 Jupyter Notebook 依序執行各 cell：

```bash
jupyter FashionMNIST_CNN_TransferLearning
```

