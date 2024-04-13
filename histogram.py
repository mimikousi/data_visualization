import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages

# ファイルの読み込み
df = pd.read_csv('debutanizer_data.csv')

#pdfで保存するためpdfインスタンスを作成
pdf = PdfPages('histogram_debutanizer.pdf')

# binの数
num_bins = 20

# 列ごとにヒストグラムを作成
for col in df.columns:
    # グラフの設定
    fig, ax = plt.subplots(figsize=(8, 5))
    # ヒストグラムの表示(density=Trueで密度表示)
    ax.hist(df[col], bins=num_bins, density=True, alpha=0.7, label='Histogram')
    # 正規分布のパラメータの推定
    mu, sigma = norm.fit(df[col])
    # 正規分布曲線のx値の範囲を決定
    x = np.linspace(df[col].min(), df[col].max(), 100)
    # 正規分布曲線の描画
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Normal Distribution')
    ax.set_title(col)
    ax.set_xlabel(f'{col}_value')
    ax.set_ylabel('Frequency')
    ax.legend()
    # グラフを保存
    fig.savefig(f'{col}_histogram.png')
    pdf.savefig(fig)
    plt.close(fig)
#close処理
pdf.close()

# スタージェスの公式
#pdfで保存するためpdfインスタンスを作成
pdf = PdfPages('histogram_sturges_debutanizer.pdf')

# 列ごとにヒストグラムを作成
for col in df.columns:
    # グラフの設定
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # ヒストグラムのbinの数を計算（スタージェスの公式）
    num_bins = int(np.log2(df[col].shape[0]) + 1)
    # ヒストグラムの表示(density=Trueで密度表示)
    ax.hist(df[col], bins=num_bins, density=True, alpha=0.7, label='Histogram')
    # ビン数を表示
    ax.text(0.5, 0.95, f'Number of bins = {num_bins}', transform=ax.transAxes, ha='center')
    # 正規分布のパラメータの推定
    mu, sigma = norm.fit(df[col])
    # 正規分布曲線のx値の範囲を決定
    x = np.linspace(df[col].min(), df[col].max(), 100)
    # 正規分布曲線の描画
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Normal Distribution')
    ax.set_title('sturges')
    ax.set_xlabel(f'{col}_value')
    ax.set_ylabel('Frequency')
    ax.legend()
    # グラフを保存
    fig.savefig(f'{col}_histogram_sturges.png')
    pdf.savefig(fig)
    plt.close(fig)
#close処理
pdf.close()

#簡易手法
#pdfで保存するためpdfインスタンスを作成
pdf = PdfPages('histogram_root_debutanizer.pdf')

# 列ごとにヒストグラムを作成
for col in df.columns:
    # グラフの設定
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # ヒストグラムのbinの数を計算（簡易手法：サンプル数の0.5乗）
    num_bins = int(df[col].shape[0] ** 0.5)
    # ヒストグラムの表示(density=Trueで密度表示)
    ax.hist(df[col], bins=num_bins, density=True, alpha=0.7, label='Histogram')
    # ビン数を表示
    ax.text(0.5, 0.95, f'Number of bins = {num_bins}', transform=ax.transAxes, ha='center')
    # 正規分布のパラメータの推定
    mu, sigma = norm.fit(df[col])
    # 正規分布曲線のx値の範囲を決定
    x = np.linspace(df[col].min(), df[col].max(), 100)
    # 正規分布曲線の描画
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Normal Distribution')
    ax.set_title('root')
    ax.set_xlabel(f'{col}_value')
    ax.set_ylabel('Frequency')
    ax.legend()
    # グラフを保存
    fig.savefig(f'{col}_histogram_root.png')
    pdf.savefig(fig)
    plt.close(fig)
#close処理
pdf.close()