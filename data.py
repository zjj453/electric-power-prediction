import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_and_process_weather(weather_path='MENSQ_01_previous-1950-2024.csv'):
    """
    加载并处理月度天气数据
    """
    # 读取天气数据（分号分隔）
    weather_df = pd.read_csv(weather_path, sep=';')

    # 筛选时间范围：200612 到 201011
    weather_df = weather_df[(weather_df['AAAAMM'] >= 200612) & (weather_df['AAAAMM'] <= 201011)]

    # 提取关键列
    weather_cols = ['AAAAMM', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU', 'TX', 'TN']
    weather_df = weather_df[weather_cols].copy()

    # 对所有气象站取平均值（因为电力数据来自法国全国/巴黎地区）
    weather_df = weather_df.groupby('AAAAMM').mean().reset_index()

    # 重命名列以便理解
    weather_df = weather_df.rename(columns={
        'RR': 'monthly_rainfall',
        'NBJRR1': 'rain_days_1mm',
        'NBJRR5': 'rain_days_5mm',
        'NBJRR10': 'rain_days_10mm',
        'NBJBROU': 'fog_days',
        'TX': 'monthly_max_temp',
        'TN': 'monthly_min_temp'
    })

    # 从AAAAMM提取年份和月份
    weather_df['year'] = weather_df['AAAAMM'].astype(str).str[:4].astype(int)
    weather_df['month'] = weather_df['AAAAMM'].astype(str).str[4:6].astype(int)

    return weather_df


def merge_weather_to_daily(daily_df, weather_df):
    """
    将月度天气数据合并到日度电力数据
    """
    # 创建合并键（年份-月份）
    daily_df = daily_df.copy()

    # 合并数据
    merged_df = pd.merge(
        daily_df,
        weather_df.drop('AAAAMM', axis=1),
        on=['year', 'month'],
        how='left'
    )

    return merged_df

def preprocess_power_data(file_path, output_path=None, weather_path='MENSQ_01_previous-1950-2024.csv'):
    """
    对家庭电力消耗数据进行预处理，将分钟级数据聚合为日度数据，并合并天气数据

    Parameters:
    -----------
    file_path : str
        原始数据文件路径（支持 .txt 或 .csv 格式）
    output_path : str, optional
        处理后数据的保存路径
    weather_path : str, optional
        天气数据文件路径

    Returns:
    --------
    df_daily : DataFrame
        日度聚合后的数据框
    """

    # 0. 加载并处理天气数据
    if weather_path and os.path.exists(weather_path):
        print(">>> 加载并处理天气数据")
        weather_df = load_and_process_weather(weather_path)
        print(f"天气数据时间范围: {weather_df['AAAAMM'].min()} - {weather_df['AAAAMM'].max()}")
        print(f"天气特征列: {[col for col in weather_df.columns if col not in ['AAAAMM', 'year', 'month']]}")
    else:
        weather_df = None
        print("警告: 未找到天气数据文件，将不合并天气特征")

    # 1. 读取原始数据
    print(f"正在读取原始数据文件: {file_path}")

    # 尝试不同参数读取 .txt 文件
    try:
        # 首先尝试用分号分隔，并处理缺失值（用 ? 表示）
        df = pd.read_csv(
            file_path,
            sep=';',
            na_values=['?'],
            low_memory=False,
            encoding='utf-8'
        )
        print(f"成功读取数据，共 {len(df)} 行记录")
    except Exception as e:
        print(f"读取失败: {e}")
        # 如果失败，尝试其他编码或分隔符
        try:
            df = pd.read_csv(
                file_path,
                sep=';',
                na_values=['?'],
                low_memory=False,
                encoding='latin1'
            )
            print(f"使用 latin1 编码成功读取数据，共 {len(df)} 行记录")
        except Exception as e2:
            print(f"再次尝试失败: {e2}")
            raise

    # 显示前几行数据以确认格式
    print("\n数据前5行预览:")
    print(df.head())
    print(f"\n数据列名: {df.columns.tolist()}")

    # 2. 合并日期和时间列
    print("\n正在处理时间信息...")

    # 检查列名（可能大小写不同）
    date_col = None
    time_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
        if 'time' in col.lower():
            time_col = col

    if date_col is None or time_col is None:
        print("警告: 未找到Date或Time列，请检查列名")
        print(f"当前列名: {df.columns.tolist()}")
        # 尝试使用第一列和第二列
        date_col = df.columns[0]
        time_col = df.columns[1]
        print(f"使用第一列 '{date_col}' 作为日期，第二列 '{time_col}' 作为时间")

    # 合并日期和时间
    df['datetime'] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str),
        format='%d/%m/%Y %H:%M:%S'
    )

    # 提取日期用于分组
    df['date'] = df['datetime'].dt.date

    print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"总天数: {df['date'].nunique()}")

    # 3. 按天聚合数据
    print("\n正在按天聚合数据...")

    # 获取数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除自动生成的列
    exclude_cols = ['date', 'datetime']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # 定义聚合方式
    agg_dict = {}
    for col in numeric_cols:
        if col in ['Global_active_power', 'Global_reactive_power',
                   'Sub_metering_1', 'Sub_metering_2']:
            agg_dict[col] = 'sum'
        elif col in ['Voltage', 'Global_intensity']:
            agg_dict[col] = 'mean'
        else:
            # 其他列默认求平均
            agg_dict[col] = 'mean'

    print(f"聚合规则: {agg_dict}")

    # 执行聚合
    df_daily = df.groupby('date').agg(agg_dict).reset_index()

    # 4. 计算 Sub_metering_3 的日总和
    if 'Sub_metering_3' in df.columns:
        sub3_daily = df.groupby('date')['Sub_metering_3'].sum().reset_index()
        sub3_daily.columns = ['date', 'Sub_metering_3_sum']
        df_daily = df_daily.merge(sub3_daily, on='date', how='left')
    else:
        print("警告: 未找到 Sub_metering_3 列")
        df_daily['Sub_metering_3_sum'] = 0

    # 计算每天的实际分钟数（数据点数量）
    minutes_per_day = df.groupby('date').size().reset_index(name='minutes_count')
    df_daily = df_daily.merge(minutes_per_day, on='date')

    # 计算剩余功率
    # Sub_metering_remainder = (global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3)
    # 注意：global_active_power 是按天求和后的值（单位：kW）
    # 转换为每天的总 Wh：kW * 1000 * 24（如果数据完整的话）
    # 更精确：用实际分钟数计算
    df_daily['Sub_metering_remainder'] = (
        df_daily['Global_active_power'] * 1000 / 60 
        - df_daily['Sub_metering_1']
        - df_daily['Sub_metering_2']
        - df_daily['Sub_metering_3_sum']
    )

    # 删除辅助列
    df_daily = df_daily.drop(['minutes_count', 'Sub_metering_3_sum'], axis=1)

    # 5. 处理缺失值
    print(f"\n处理缺失值前数据形状: {df_daily.shape}")
    missing_stats = df_daily.isnull().sum()
    if missing_stats.sum() > 0:
        print(f"缺失值统计:\n{missing_stats[missing_stats > 0]}")
    else:
        print("数据无缺失值")

    # 对于缺失值，使用前向填充 + 后向填充
    for col in df_daily.columns:
        if df_daily[col].dtype in ['float64', 'int64']:
            df_daily[col] = df_daily[col].fillna(method='ffill').fillna(method='bfill')
            # 如果还有缺失，用均值填充
            df_daily[col] = df_daily[col].fillna(df_daily[col].mean())

    print(f"处理缺失值后数据形状: {df_daily.shape}")

    # 6. 添加时间特征（可选，但有助于模型学习）
    print("\n正在添加时间特征...")
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['year'] = df_daily['date'].dt.year
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['day'] = df_daily['date'].dt.day
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_daily['day_of_year'] = df_daily['date'].dt.dayofyear
    df_daily['quarter'] = df_daily['date'].dt.quarter

    # 7. 合并天气数据
    if weather_df is not None:
        print("\n>>> 合并天气数据到电力数据")
        df_daily = merge_weather_to_daily(df_daily, weather_df)
        print(f"合并后数据形状: {df_daily.shape}")

    # 8. 保存处理后的数据
    if output_path:
        df_daily.to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存至: {output_path}")

    # 9. 打印数据摘要
    print("\n" + "="*60)
    print("数据预处理完成！数据摘要：")
    print("="*60)
    print(f"日期范围: {df_daily['date'].min()} 到 {df_daily['date'].max()}")
    print(f"总天数: {len(df_daily)}")
    print(f"\n特征列 ({len(df_daily.columns)} 列):")
    print(df_daily.columns.tolist())
    print(f"\n各列统计信息:")
    print(df_daily.describe())

    return df_daily


def create_sequences(data, target_col='Global_active_power', input_days=90, output_days=90):
    """
    创建时间序列样本（用于多变量预测）

    Parameters:
    -----------
    data : DataFrame
        时间序列数据
    target_col : str
        目标变量列名
    input_days : int
        输入序列长度
    output_days : int
        预测序列长度

    Returns:
    --------
    X : array
        输入序列（包含所有特征）
    y : array
        目标序列
    feature_cols : list
        特征列名列表
    """
    X, y = [], []

    # 获取特征列（排除日期和时间相关列）
    exclude_cols = ['date', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter']
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    # 确保目标列在特征中
    if target_col not in feature_cols:
        feature_cols.append(target_col)

    # 检查是否有天气特征
    weather_feature_cols = ['monthly_rainfall', 'rain_days_1mm', 'rain_days_5mm',
                           'rain_days_10mm', 'fog_days', 'monthly_max_temp', 'monthly_min_temp']
    has_weather = any(col in data.columns for col in weather_feature_cols)
    if has_weather:
        print(f"检测到天气特征: {[col for col in weather_feature_cols if col in data.columns]}")

    print(f"\n使用特征列: {feature_cols}")
    print(f"目标列: {target_col}")

    # 提取特征值和目标值
    features = data[feature_cols].values
    target = data[target_col].values

    total_len = len(data)
    print(f"数据总长度: {total_len}")
    print(f"输入窗口: {input_days}, 输出窗口: {output_days}")

    # 创建样本
    for i in range(total_len - input_days - output_days + 1):
        X.append(features[i:i+input_days])
        y.append(target[i+input_days:i+input_days+output_days])

    X = np.array(X)
    y = np.array(y)

    print(f"生成的样本数: {len(X)}")
    print(f"X shape: {X.shape} (样本数, 时间步, 特征数)")
    print(f"y shape: {y.shape} (样本数, 预测步数)")

    return X, y, feature_cols


def prepare_data_for_modeling(train_file, test_file, input_days=90, output_days=90):
    """
    完整的数据准备流程
    """
    print("="*60)
    print("开始数据准备流程")
    print("="*60)
    
    # 1. 处理训练数据
    print("\n>>> 处理训练数据")
    train_daily = preprocess_power_data(train_file, 'train_daily.csv')
    
    # 2. 处理测试数据
    print("\n>>> 处理测试数据")
    test_daily = preprocess_power_data(test_file, 'test_daily.csv')
    
    # 3. 创建序列
    print("\n>>> 创建训练序列")
    X_train, y_train, feature_cols = create_sequences(
        train_daily, 
        input_days=input_days, 
        output_days=output_days
    )
    
    print("\n>>> 创建测试序列")
    X_test, y_test, _ = create_sequences(
        test_daily, 
        input_days=input_days, 
        output_days=output_days
    )
    
    # 4. 保存处理后的数据
    print("\n>>> 保存处理结果")
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    
    # 保存特征列名
    with open('feature_cols.txt', 'w') as f:
        f.write(','.join(feature_cols))
    
    print("\n" + "="*60)
    print("数据准备完成！")
    print("="*60)
    print(f"训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试集: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"\n特征列 ({len(feature_cols)} 个):")
    print(feature_cols)
    print("\n所有文件已保存。")
    
    return X_train, y_train, X_test, y_test, feature_cols


if __name__ == "__main__":
    # 使用示例
    # 注意：请根据实际文件路径修改
    train_file = 'household_power_consumption.txt'  # 如果文件在当前目录
    
    # 如果文件在其他位置，请使用完整路径
    # train_file = '/path/to/household_power_consumption.txt'
    
    # 准备数据
    try:
        X_train, y_train, X_test, y_test, feature_cols = prepare_data_for_modeling(
            train_file=train_file,
            test_file=train_file,  # 如果只有训练文件，先用同一个
            input_days=90,
            output_days=90  # 短期预测
        )
        
        # 如果要准备长期预测数据
        # X_train_long, y_train_long, X_test_long, y_test_long, _ = prepare_data_for_modeling(
        #     train_file=train_file,
        #     test_file=train_file,
        #     input_days=90,
        #     output_days=365  # 长期预测
        # )
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{train_file}'")
        print("请确认文件路径是否正确。")
        print("如果文件在其他位置，请修改 train_file 变量。")