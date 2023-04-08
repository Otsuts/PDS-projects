# <center> 这是数据科学基础课的作业集合

# Project1：

usage:

```
python main.py --C --seed --reduce_method --reduced_dim
```
分别是：SVM 的C， 随机种子， 使用的降维方法，需要将到的目标维数


# Project2:
本次作业要求我们用KNN分类上一次的数据集，采用不同的距离度量方式，并且采用metric learning进一步提升效果

采用网格式搜索超参，直接运行：
```
python run_script.py
```
具体参数：
```
usage: main.py [-h] [--method METHOD] [--k_neighbors K_NEIGHBORS]
               [--seed SEED] [--log_path LOG_PATH]
               [--metric_learning {NCA,LFDA,MLKR,None}]

optional arguments:
  -h, --help            show this help message and exit
  --method METHOD
  --k_neighbors K_NEIGHBORS
  --seed SEED
  --log_path LOG_PATH
  --metric_learning {NCA,LFDA,MLKR,None}
  ```