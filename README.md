## 数据集下载

```powershell
wget -P data/raw https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz
```

## 下载依赖

### 方法1

```powershell

# 选择如下配置，不容易出问题:PyTorch 2.0.1 + CUDA 11.8 镜像，Python 3.10 开箱自带
# 下载下面包即可
pip install sentence-transformers shap streamlit plotly wordcloud nltk packaging transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 方法2
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