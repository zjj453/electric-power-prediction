import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以保证可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==================== 数据加载和预处理 ====================
def load_and_prepare_data(train_daily_path='train_daily_with_weather.csv', test_daily_path=None, split_date='2009-01-01'):
    """
    加载日度数据（已包含天气特征）并准备训练和测试集
    按时间顺序划分：用较早的数据训练，较晚的数据测试
    """
    # 加载数据
    df = pd.read_csv(train_daily_path)
    df['date'] = pd.to_datetime(df['date'])

    if test_daily_path and os.path.exists(test_daily_path):
        # 如果有单独的测试集文件
        test_df = pd.read_csv(test_daily_path)
        test_df['date'] = pd.to_datetime(test_df['date'])
        train_df = df
    else:
        # 按时间划分训练集和测试集
        split_date = pd.to_datetime(split_date)
        train_df = df[df['date'] < split_date].copy()
        test_df = df[df['date'] >= split_date].copy()

    print(f"训练集时间范围: {train_df['date'].min()} 到 {train_df['date'].max()}")
    print(f"测试集时间范围: {test_df['date'].min()} 到 {test_df['date'].max()}")

    return train_df, test_df


def create_sequences_multivariate(data, feature_cols, target_col='Global_active_power',
                                   input_days=90, output_days=90):
    """
    创建多变量时间序列样本
    """
    X, y = [], []
    features = data[feature_cols].values
    target = data[target_col].values

    for i in range(len(data) - input_days - output_days + 1):
        X.append(features[i:i+input_days])
        y.append(target[i+input_days:i+input_days+output_days])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data_for_training(train_df, test_df, feature_cols, input_days=90, output_days=90):
    """
    完整的数据准备：创建序列、标准化
    """
    # 创建序列
    X_train, y_train = create_sequences_multivariate(
        train_df, feature_cols, input_days=input_days, output_days=output_days
    )
    X_test, y_test = create_sequences_multivariate(
        test_df, feature_cols, input_days=input_days, output_days=output_days
    )

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 重塑为2D进行标准化
    n_samples_train, n_steps, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(n_samples_train, n_steps, n_features)

    n_samples_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(n_samples_test, n_steps, n_features)

    # 标准化目标值
    y_train_flat = y_train.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_flat).reshape(y_train.shape)

    y_test_flat = y_test.reshape(-1, 1)
    y_test_scaled = scaler_y.transform(y_test_flat).reshape(y_test.shape)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y


# ==================== LSTM模型 ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=90, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 只取最后一个时间步的输出
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


# ==================== Transformer模型 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, output_size=90, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 使用最后一个时间步的输出
        out = x[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ==================== CNN-LSTM-Transformer 混合模型 ====================
class CNNLSTMTransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, output_size=90, dropout=0.1):
        super(CNNLSTMTransformerModel, self).__init__()

        # 第一层：1D CNN - 提取局部时序特征
        self.conv_layers = nn.Sequential(
            # 第一个卷积块：7天窗口
            nn.Conv1d(input_size, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(dropout),

            # 第二个卷积块：14天窗口
            nn.Conv1d(64, 128, kernel_size=14, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(dropout),

            # 第三个卷积层：输出到d_model维度
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 第二层：双层 LSTM - 捕获中短期时序依赖
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2,
                           batch_first=True, dropout=dropout)

        # 第三层：多头 Transformer 编码器 - 建模全局长距离依赖
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, features)
        # CNN 需要 (batch_size, features, seq_len)
        x = x.transpose(1, 2)

        # 第一层：CNN 提取局部特征
        x = self.conv_layers(x)

        # 转回 (batch_size, seq_len, features)
        x = x.transpose(1, 2)

        # 第二层：LSTM 捕获中短期依赖
        lstm_out, _ = self.lstm(x)

        # 第三层：Transformer 建模长距离依赖
        x = self.pos_encoder(lstm_out)
        x = self.transformer_encoder(x)

        # 使用最后一个时间步的输出
        out = x[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
    """
    训练模型，使用早停策略
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses


def run_experiment(model_class, X_train, y_train, X_val, y_val,
                   model_params, epochs=100, batch_size=32, lr=0.001):
    """
    运行单次实验
    """
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = model_class(**model_params)

    # 训练
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr
    )

    return model, train_losses, val_losses


# ==================== 评估函数 ====================
def evaluate_model(model, X_test, y_test, scaler_y):
    """
    评估模型性能
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    # 反标准化
    predictions_flat = predictions.reshape(-1, 1)
    predictions_original = scaler_y.inverse_transform(predictions_flat).reshape(predictions.shape)

    y_test_flat = y_test.reshape(-1, 1)
    y_test_original = scaler_y.inverse_transform(y_test_flat).reshape(y_test.shape)

    # 计算MSE和MAE（对所有预测步长）
    mse = mean_squared_error(y_test_original.flatten(), predictions_original.flatten())
    mae = mean_absolute_error(y_test_original.flatten(), predictions_original.flatten())

    return predictions_original, y_test_original, mse, mae


def run_multiple_experiments(model_class, X_train, y_train, X_test, y_test, scaler_y,
                             model_params, n_runs=5, epochs=100, batch_size=32, lr=0.001):
    """
    运行多轮实验并返回统计结果
    """
    mse_list = []
    mae_list = []
    predictions_list = []

    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"实验第 {run+1}/{n_runs} 轮")
        print('='*50)

        # 划分验证集
        val_size = int(len(X_train) * 0.2)
        X_train_fold = X_train[:-val_size]
        y_train_fold = y_train[:-val_size]
        X_val_fold = X_train[-val_size:]
        y_val_fold = y_train[-val_size:]

        # 训练
        model, _, _ = run_experiment(
            model_class, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            model_params, epochs=epochs, batch_size=batch_size, lr=lr
        )

        # 评估
        predictions, y_true, mse, mae = evaluate_model(model, X_test, y_test, scaler_y)
        mse_list.append(mse)
        mae_list.append(mae)
        predictions_list.append(predictions)

        print(f"Run {run+1}: MSE={mse:.4f}, MAE={mae:.4f}")

    # 计算统计量
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)

    return {
        'mse_mean': mse_mean, 'mse_std': mse_std,
        'mae_mean': mae_mean, 'mae_std': mae_std,
        'mse_list': mse_list, 'mae_list': mae_list,
        'predictions_list': predictions_list,
        'y_true': y_true
    }


# ==================== 绘图函数 ====================
def plot_predictions(y_true, predictions, title="预测对比"):
    """
    绘制预测结果对比图
    """
    plt.figure(figsize=(15, 6))

    # 选择第一个样本进行可视化
    sample_idx = 0
    y_true_sample = y_true[sample_idx]
    pred_sample = predictions[sample_idx]

    plt.plot(y_true_sample, label='真实值', linewidth=2, color='blue')
    plt.plot(pred_sample, label='预测值', linewidth=2, color='red', linestyle='--')

    plt.xlabel('预测天数', fontsize=12)
    plt.ylabel('总有功功率 (kW)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_predictions(y_true, all_predictions, title="多轮预测对比"):
    """
    绘制所有轮次的预测结果
    """
    plt.figure(figsize=(15, 6))

    sample_idx = 0
    y_true_sample = y_true[sample_idx]

    plt.plot(y_true_sample, label='真实值', linewidth=2, color='blue')

    # 绘制所有预测
    for i, pred in enumerate(all_predictions):
        plt.plot(pred[sample_idx], alpha=0.3, color='red', linewidth=1)

    # 绘制平均预测
    avg_pred = np.mean(all_predictions, axis=0)[sample_idx]
    plt.plot(avg_pred, label='平均预测', linewidth=2, color='orange', linestyle='--')

    plt.xlabel('预测天数', fontsize=12)
    plt.ylabel('总有功功率 (kW)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================
def run_prediction_experiment(train_df, test_df, feature_cols, input_days, output_days, experiment_name):
    """
    运行一组预测实验（短期或长期）
    """
    print(f"\n{'='*60}")
    print(f"开始 {experiment_name} 实验")
    print(f"{'='*60}")

    # 准备数据
    print(f"\n>>> 准备训练数据 (输入{input_days}天, 输出{output_days}天)")
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data_for_training(
        train_df, test_df, feature_cols, input_days=input_days, output_days=output_days
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # LSTM实验
    print(f"\n{'-'*60}")
    print(f"{experiment_name} - LSTM 模型实验 (5轮)")
    print(f"{'-'*60}")

    lstm_params = {
        'input_size': len(feature_cols),
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': output_days,
        'dropout': 0.2
    }

    lstm_results = run_multiple_experiments(
        LSTMModel, X_train, y_train, X_test, y_test, scaler_y,
        lstm_params, n_runs=5, epochs=100, batch_size=32, lr=0.001
    )

    print(f"\nLSTM 实验结果:")
    print(f"MSE: {lstm_results['mse_mean']:.4f} ± {lstm_results['mse_std']:.4f}")
    print(f"MAE: {lstm_results['mae_mean']:.4f} ± {lstm_results['mae_std']:.4f}")

    # Transformer实验
    print(f"\n{'-'*60}")
    print(f"{experiment_name} - Transformer 模型实验 (5轮)")
    print(f"{'-'*60}")

    transformer_params = {
        'input_size': len(feature_cols),
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'output_size': output_days,
        'dropout': 0.1
    }

    transformer_results = run_multiple_experiments(
        TransformerModel, X_train, y_train, X_test, y_test, scaler_y,
        transformer_params, n_runs=5, epochs=100, batch_size=32, lr=0.001
    )

    print(f"\nTransformer 实验结果:")
    print(f"MSE: {transformer_results['mse_mean']:.4f} ± {transformer_results['mse_std']:.4f}")
    print(f"MAE: {transformer_results['mae_mean']:.4f} ± {transformer_results['mae_std']:.4f}")

    # CNN-LSTM-Transformer 混合模型实验
    print(f"\n{'-'*60}")
    print(f"{experiment_name} - CNN-LSTM-Transformer 混合模型实验 (5轮)")
    print(f"{'-'*60}")

    cnn_lstm_transformer_params = {
        'input_size': len(feature_cols),
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'output_size': output_days,
        'dropout': 0.1
    }

    cnn_lstm_transformer_results = run_multiple_experiments(
        CNNLSTMTransformerModel, X_train, y_train, X_test, y_test, scaler_y,
        cnn_lstm_transformer_params, n_runs=5, epochs=100, batch_size=32, lr=0.001
    )

    print(f"\nCNN-LSTM-Transformer 混合模型实验结果:")
    print(f"MSE: {cnn_lstm_transformer_results['mse_mean']:.4f} ± {cnn_lstm_transformer_results['mse_std']:.4f}")
    print(f"MAE: {cnn_lstm_transformer_results['mae_mean']:.4f} ± {cnn_lstm_transformer_results['mae_std']:.4f}")

    # 可视化对比
    print(f"\n>>> 绘制 {experiment_name} 预测对比图")

    # LSTM预测图
    plot_predictions(
        lstm_results['y_true'],
        lstm_results['predictions_list'][0],
        title=f"LSTM 预测 vs 真实值 ({experiment_name})"
    )

    # Transformer预测图
    plot_predictions(
        transformer_results['y_true'],
        transformer_results['predictions_list'][0],
        title=f"Transformer 预测 vs 真实值 ({experiment_name})"
    )

    # CNN-LSTM-Transformer预测图
    plot_predictions(
        cnn_lstm_transformer_results['y_true'],
        cnn_lstm_transformer_results['predictions_list'][0],
        title=f"CNN-LSTM-Transformer 预测 vs 真实值 ({experiment_name})"
    )

    # 多轮预测对比
    plot_all_predictions(
        lstm_results['y_true'],
        lstm_results['predictions_list'],
        title=f"LSTM 5轮预测对比 ({experiment_name})"
    )

    plot_all_predictions(
        transformer_results['y_true'],
        transformer_results['predictions_list'],
        title=f"Transformer 5轮预测对比 ({experiment_name})"
    )

    plot_all_predictions(
        cnn_lstm_transformer_results['y_true'],
        cnn_lstm_transformer_results['predictions_list'],
        title=f"CNN-LSTM-Transformer 5轮预测对比 ({experiment_name})"
    )

    return {
        'lstm': lstm_results,
        'transformer': transformer_results,
        'cnn_lstm_transformer': cnn_lstm_transformer_results
    }


def main():
    print("="*60)
    print("家庭电力消耗预测 - LSTM vs Transformer vs CNN-LSTM-Transformer (含天气特征)")
    print("="*60)

    # 1. 加载数据
    print("\n>>> 加载数据")
    # 使用一个文件，按时间划分（2009-01-01之前为训练集，之后为测试集）
    train_df, test_df = load_and_prepare_data(
        'train_daily_with_weather.csv',
        None,  # 不使用单独的测试文件
        split_date='2009-01-01'
    )

    # 选择特征（排除日期相关列，但保留天气特征）
    exclude_cols = ['date', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter']
    weather_feature_cols = ['monthly_rainfall', 'rain_days_1mm', 'rain_days_5mm',
                           'rain_days_10mm', 'fog_days', 'monthly_max_temp', 'monthly_min_temp']

    # 原始电力特征 + 天气特征
    power_feature_cols = [col for col in train_df.columns if col not in exclude_cols and col not in weather_feature_cols]
    feature_cols = power_feature_cols + weather_feature_cols
    target_col = 'Global_active_power'

    print(f"电力特征: {power_feature_cols}")
    print(f"天气特征: {weather_feature_cols}")
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")

    # 2. 短期预测实验 (90->90)
    short_term_results = run_prediction_experiment(
        train_df, test_df, feature_cols,
        input_days=90, output_days=90,
        experiment_name="短期预测 (90->90)"
    )

    # 3. 长期预测实验 (90->365)
    long_term_results = run_prediction_experiment(
        train_df, test_df, feature_cols,
        input_days=90, output_days=365,
        experiment_name="长期预测 (90->365)"
    )

    # 4. 汇总所有结果
    print("\n" + "="*60)
    print("实验结果汇总 (含天气特征)")
    print("="*60)

    summary_data = {
        '预测类型': ['短期 (90->90)', '短期 (90->90)', '短期 (90->90)', '长期 (90->365)', '长期 (90->365)', '长期 (90->365)'],
        '模型': ['LSTM', 'Transformer', 'CNN-LSTM-Transformer', 'LSTM', 'Transformer', 'CNN-LSTM-Transformer'],
        'MSE Mean': [
            short_term_results['lstm']['mse_mean'],
            short_term_results['transformer']['mse_mean'],
            short_term_results['cnn_lstm_transformer']['mse_mean'],
            long_term_results['lstm']['mse_mean'],
            long_term_results['transformer']['mse_mean'],
            long_term_results['cnn_lstm_transformer']['mse_mean']
        ],
        'MSE Std': [
            short_term_results['lstm']['mse_std'],
            short_term_results['transformer']['mse_std'],
            short_term_results['cnn_lstm_transformer']['mse_std'],
            long_term_results['lstm']['mse_std'],
            long_term_results['transformer']['mse_std'],
            long_term_results['cnn_lstm_transformer']['mse_std']
        ],
        'MAE Mean': [
            short_term_results['lstm']['mae_mean'],
            short_term_results['transformer']['mae_mean'],
            short_term_results['cnn_lstm_transformer']['mae_mean'],
            long_term_results['lstm']['mae_mean'],
            long_term_results['transformer']['mae_mean'],
            long_term_results['cnn_lstm_transformer']['mae_mean']
        ],
        'MAE Std': [
            short_term_results['lstm']['mae_std'],
            short_term_results['transformer']['mae_std'],
            short_term_results['cnn_lstm_transformer']['mae_std'],
            long_term_results['lstm']['mae_std'],
            long_term_results['transformer']['mae_std'],
            long_term_results['cnn_lstm_transformer']['mae_std']
        ]
    }

    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))

    # 保存结果到CSV
    results_df.to_csv('experiment_results_summary.csv', index=False)
    print("\n结果已保存到 experiment_results_summary.csv")

    return short_term_results, long_term_results



if __name__ == "__main__":
    short_term_results, long_term_results = main()
