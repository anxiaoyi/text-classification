# text-classification

## 北邮数据挖掘课王晓茹文本分类作业一

### 环境

``` Bash
pip install jieba
pip install scikit-learn
```

在 `src/utils/data_mining/data/source-data` 目录下，要确保存在以下几个分类目录(教育、能源、金融、房产、科技、健康、军事、体育、文化、娱乐)，并且每个目录下面要放 10W 篇以 txt 结尾的相应的文章。

```
source-data/
├── edu
	├── a.txt
	├── b.txt
	├── c.txt
	├── ...
├── energy
├── finance
├── house
├── IT
├── jk
├── mil
├── sports
├── wh
└── yl
```

### 运行

``` Python
cd src/utils/data_mining/
python3 fenci_fixed_data.py
```
