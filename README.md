# 家庭电力消耗预测

基于深度学习的家庭电力消耗预测项目，对比三种模型：LSTM、Transformer和CNN-LSTM-Transformer混合模型。

## 📊 项目简介

使用历史电力数据+天气特征，预测未来电力消耗：
- **短期预测**: 90天历史 → 90天预测
- **长期预测**: 90天历史 → 365天预测

## 🛠️ 模型架构

### LSTM
- 双层LSTM结构
- 稳定性最佳，适合生产环境

### Transformer
- 位置编码 + 多头注意力
- 短期预测表现最佳

### CNN-LSTM-Transformer (混合模型) ⭐
- **CNN层**: 提取7-14天局部用电模式
- **LSTM层**: 捕获中短期时序依赖
- **Transformer层**: 建模跨月份、跨季节长期规律
- **长期预测表现最佳**

## 📁 文件说明

- `data.py`: 数据预处理和天气数据合并
- `model_training.py`: 模型训练和评估
- `merge_weather.py`: 天气数据合并脚本
- `requirements.txt`: Python依赖包（需要自行生成）

## 🎯 实验结果

### 短期预测 (90->90)
| 模型 | MSE | MAE |
|------|-----|-----|
| LSTM | 291,537 | 415.17 |
| Transformer | 237,276 | 383.77 |
| CNN-LSTM-Transformer | 287,951 | 412.42 |

### 长期预测 (90->365)
| 模型 | MSE | MAE |
|------|-----|-----|
| LSTM | 233,484 | 366.66 |
| Transformer | 247,993 | 380.15 |
| CNN-LSTM-Transformer | 215,786 | 347.30 |

## 🚀 使用方法

```bash
# 1. 安装依赖
pip install numpy pandas torch scikit-learn matplotlib

# 2. 运行训练
python model_training.py
```

## 📝 数据来源

- 电力数据：家庭电力消耗数据集
- 天气数据：MENSQ气象数据集

## 📄 许可证

MIT License
