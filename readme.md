## 数据集下载

wget -P data/raw https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz

## 下载依赖

```powershell
# 创建一个 Python 3.10 新环境
conda create -n torch python=3.10 -y

# 激活环境
conda activate torch

pip install -r requirements.txt

# 仅zhx使用
pip install -r requirements-gpu-compat.txt
```

## 数据预处理

```powershell
python src/preprocess.py
```