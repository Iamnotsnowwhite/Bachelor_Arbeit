import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# 读取CSV文件
file_path = "SW1783_mc_dropout+mean_variance_target.csv"
df = pd.read_csv(file_path)

# 计算标准差
df['std_epi'] = np.sqrt(df['var_epi'])
df['std_ale'] = np.sqrt(df['var_ale'])

# 创建简化图表
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 主要饼图：不确定性来源比例
total_var_epi = df['var_epi'].sum()
total_var_ale = df['var_ale'].sum()
total_variance = total_var_epi + total_var_ale

axes[0, 0].pie([total_var_epi, total_var_ale],
               labels=['Epistemic\n(Unsicherheit am Model)', 'Aleatoric\n(Unsicherheit an Daten)'],
               autopct='%1.1f%%',
               startangle=90,
               colors=['#FF6B6B', '#4ECDC4'],
               explode=(0.05, 0))
axes[0, 0].set_title('Uncertainty Sources Distribution', fontweight='bold')

# 2. 简单的箱线图比较
uncertainty_data = [df['std_epi'], df['std_ale']]
box = axes[0, 1].boxplot(uncertainty_data, patch_artist=True,
                        labels=['Epistemic', 'Aleatoric'])
colors = ['#FF6B6B', '#4ECDC4']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 1].set_ylabel('Standard Deviation')
axes[0, 1].set_title('Uncertainty Comparison', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. 散点图：两种不确定性的关系
scatter = axes[1, 0].scatter(df['std_epi'], df['std_ale'], 
                           alpha=0.7, s=60, c='#45B7D1')
axes[1, 0].plot([0, df['std_epi'].max()], [0, df['std_epi'].max()], 
               'r--', alpha=0.7, label='Equal Line')
axes[1, 0].set_xlabel('Epistemic Uncertainty (Std)')
axes[1, 0].set_ylabel('Aleatoric Uncertainty (Std)')
axes[1, 0].set_title('Relationship Between Uncertainties', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 误差与总不确定性的关系
total_std = np.sqrt(df['var_total'])
axes[1, 1].scatter(total_std, df['abs_error'], alpha=0.7, s=60, c='#F9A602')
axes[1, 1].set_xlabel('Total Uncertainty (Std)')
axes[1, 1].set_ylabel('Absolute Prediction Error')
axes[1, 1].set_title('Uncertainty vs Prediction Error', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# 计算相关系数
correlation = np.corrcoef(total_std, df['abs_error'])[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=axes[1, 1].transAxes, fontsize=12,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig('SW1783_simple_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出关键统计信息
print("=" * 50)
print("KEY UNCERTAINTY STATISTICS")
print("=" * 50)
print(f"Total samples: {len(df)}")
print(f"Epistemic uncertainty: {df['std_epi'].mean():.3f} ± {df['std_epi'].std():.3f}")
print(f"Aleatoric uncertainty: {df['std_ale'].mean():.3f} ± {df['std_ale'].std():.3f}")
print(f"Epistemic proportion: {(total_var_epi/total_variance)*100:.1f}%")
print(f"Correlation (epi vs ale): {df[['std_epi', 'std_ale']].corr().iloc[0,1]:.3f}")
print(f"Uncertainty-Error correlation: {correlation:.3f}")
print("=" * 50)