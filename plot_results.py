import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 实验结果数据
short_term_results = {
    'LSTM': {'MSE': 291537.34, 'MAE': 415.17},
    'Transformer': {'MSE': 237276.12, 'MAE': 383.77},
    'CNN-LSTM-Transformer': {'MSE': 287951.31, 'MAE': 412.42}
}

long_term_results = {
    'LSTM': {'MSE': 233484.28, 'MAE': 366.66},
    'Transformer': {'MSE': 247993.03, 'MAE': 380.15},
    'CNN-LSTM-Transformer': {'MSE': 215786.03, 'MAE': 347.30}
}

model_names = list(short_term_results.keys())
colors = ['#27ae60', '#e74c3c', '#9b59b6']


def plot_mse_comparison():
    """绘制MSE对比柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(model_names))
    width = 0.6

    # 短期预测MSE
    short_mse = [short_term_results[name]['MSE'] for name in model_names]
    bars1 = ax1.bar(x, short_mse, width, color=colors, alpha=0.8)
    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('短期预测 (90->90) - MSE对比', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11, rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=10)

    # 长期预测MSE
    long_mse = [long_term_results[name]['MSE'] for name in model_names]
    bars2 = ax2.bar(x, long_mse, width, color=colors, alpha=0.8)
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('长期预测 (90->365) - MSE对比', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=11, rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.suptitle('MSE对比 - 短期vs长期', fontsize=15, fontweight='bold', y=1.02)
    plt.savefig('MSE对比图.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_mae_comparison():
    """绘制MAE对比柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(model_names))
    width = 0.6

    # 短期预测MAE
    short_mae = [short_term_results[name]['MAE'] for name in model_names]
    bars1 = ax1.bar(x, short_mae, width, color=colors, alpha=0.8)
    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('短期预测 (90->90) - MAE对比', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11, rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    # 长期预测MAE
    long_mae = [long_term_results[name]['MAE'] for name in model_names]
    bars2 = ax2.bar(x, long_mae, width, color=colors, alpha=0.8)
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('长期预测 (90->365) - MAE对比', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=11, rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.suptitle('MAE对比 - 短期vs长期', fontsize=15, fontweight='bold', y=1.02)
    plt.savefig('MAE对比图.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_summary_table():
    """绘制汇总表格"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['模型', '短期MSE', '短期MAE', '长期MSE', '长期MAE'],
        ['LSTM', f'{short_term_results["LSTM"]["MSE"]:,.0f}', f'{short_term_results["LSTM"]["MAE"]:.2f}',
         f'{long_term_results["LSTM"]["MSE"]:,.0f}', f'{long_term_results["LSTM"]["MAE"]:.2f}'],
        ['Transformer', f'{short_term_results["Transformer"]["MSE"]:,.0f}', f'{short_term_results["Transformer"]["MAE"]:.2f}',
         f'{long_term_results["Transformer"]["MSE"]:,.0f}', f'{long_term_results["Transformer"]["MAE"]:.2f}'],
        ['CNN-LSTM-Transformer', f'{short_term_results["CNN-LSTM-Transformer"]["MSE"]:,.0f}', f'{short_term_results["CNN-LSTM-Transformer"]["MAE"]:.2f}',
         f'{long_term_results["CNN-LSTM-Transformer"]["MSE"]:,.0f}', f'{long_term_results["CNN-LSTM-Transformer"]["MAE"]:.2f}']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # 标记最佳结果
    for i in range(1, 4):
        # 短期MSE
        if i == 2:
            table[(i, 1)].set_facecolor('#e8f6f3')
        # 短期MAE
        if i == 2:
            table[(i, 2)].set_facecolor('#e8f6f3')
        # 长期MSE
        if i == 3:
            table[(i, 3)].set_facecolor('#e8f6f3')
        # 长期MAE
        if i == 3:
            table[(i, 4)].set_facecolor('#e8f6f3')

    plt.title('实验结果汇总表', fontsize=15, fontweight='bold', pad=20)
    plt.savefig('结果汇总表.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("生成可视化对比图...")
    print("\n1. 绘制MSE对比图...")
    plot_mse_comparison()

    print("2. 绘制MAE对比图...")
    plot_mae_comparison()

    print("3. 绘制汇总表格...")
    plot_summary_table()

    print("\n所有图表生成完成！已保存为PNG文件")
