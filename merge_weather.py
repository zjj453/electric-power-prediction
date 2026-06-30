import pandas as pd
import numpy as np
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


if __name__ == "__main__":
    print("="*60)
    print("合并天气数据到电力数据集")
    print("="*60)

    # 1. 加载天气数据
    print("\n>>> 加载天气数据...")
    weather_df = load_and_process_weather()
    print(f"天气数据形状: {weather_df.shape}")
    print(f"天气数据时间范围: {weather_df['AAAAMM'].min()} - {weather_df['AAAAMM'].max()}")
    weather_features = [col for col in weather_df.columns if col not in ['AAAAMM', 'year', 'month']]
    print(f"天气特征: {weather_features}")

    # 2. 加载并处理训练数据
    print("\n>>> 处理训练数据...")
    train_df = pd.read_csv('train_daily.csv')
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df = merge_weather_to_daily(train_df, weather_df)
    train_df.to_csv('train_daily_with_weather.csv', index=False)
    print(f"训练数据合并后形状: {train_df.shape}")
    train_columns = train_df.columns.tolist()
    print(f"训练数据列: {train_columns}")

    # 3. 加载并处理测试数据
    print("\n>>> 处理测试数据...")
    test_df = pd.read_csv('test_daily.csv')
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df = merge_weather_to_daily(test_df, weather_df)
    test_df.to_csv('test_daily_with_weather.csv', index=False)
    print(f"测试数据合并后形状: {test_df.shape}")

    print("\n" + "="*60)
    print("完成！生成的文件:")
    print("- train_daily_with_weather.csv")
    print("- test_daily_with_weather.csv")
    print("="*60)
