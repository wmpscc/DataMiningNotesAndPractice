
# 数据预处理
> 注意：本篇文章为读书笔记，若有版权问题，请联系我删除。
### 服装消费者数据描述：
- age:年龄
- gender:性别
- income:收入
- house:是否有房子
- store_exp:实体店消费额
- online_exp:在线消费额
- store_trans:在实体店交易次数
- online_trans:在线交易次数
- Q1~Q10:问卷的10个问题（非常不同意：1；有点不同意：2；中立/不知道：3；有点同意：4；非常同意：5）
- segment:消费者分组(价格敏感：Price；炫耀性消费：Conspicuous；质量：Quality；风格：Style)


```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
data = pd.read_csv("data/segdata.csv")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>income</th>
      <th>house</th>
      <th>store_exp</th>
      <th>online_exp</th>
      <th>store_trans</th>
      <th>online_trans</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>Q8</th>
      <th>Q9</th>
      <th>Q10</th>
      <th>segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57</td>
      <td>Female</td>
      <td>120963.400958</td>
      <td>Yes</td>
      <td>529.134363</td>
      <td>303.512475</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>Price</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>Female</td>
      <td>122008.104950</td>
      <td>Yes</td>
      <td>478.005781</td>
      <td>109.529710</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>Price</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59</td>
      <td>Male</td>
      <td>114202.295294</td>
      <td>Yes</td>
      <td>490.810731</td>
      <td>279.249582</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>Price</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>Male</td>
      <td>113616.337078</td>
      <td>Yes</td>
      <td>347.809004</td>
      <td>141.669752</td>
      <td>10</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>Price</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>Male</td>
      <td>124252.552787</td>
      <td>Yes</td>
      <td>379.625940</td>
      <td>112.237177</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>Price</td>
    </tr>
  </tbody>
</table>
</div>



## 检查数据

读入数据中的第一步，就是检查数据，看看都有哪些变量，这些变量分布如何，是不是存在错误的观测。


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 19 columns):
    age             1000 non-null int64
    gender          1000 non-null object
    income          816 non-null float64
    house           1000 non-null object
    store_exp       1000 non-null float64
    online_exp      1000 non-null float64
    store_trans     1000 non-null int64
    online_trans    1000 non-null int64
    Q1              1000 non-null int64
    Q2              1000 non-null int64
    Q3              1000 non-null int64
    Q4              1000 non-null int64
    Q5              1000 non-null int64
    Q6              1000 non-null int64
    Q7              1000 non-null int64
    Q8              1000 non-null int64
    Q9              1000 non-null int64
    Q10             1000 non-null int64
    segment         1000 non-null object
    dtypes: float64(3), int64(13), object(3)
    memory usage: 148.5+ KB



```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
      <th>store_exp</th>
      <th>online_exp</th>
      <th>store_trans</th>
      <th>online_trans</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>Q8</th>
      <th>Q9</th>
      <th>Q10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>816.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.840000</td>
      <td>113543.065222</td>
      <td>1356.850523</td>
      <td>2120.181187</td>
      <td>5.350000</td>
      <td>13.546000</td>
      <td>3.101000</td>
      <td>1.823000</td>
      <td>1.992000</td>
      <td>2.763000</td>
      <td>2.945000</td>
      <td>2.448000</td>
      <td>3.434000</td>
      <td>2.396000</td>
      <td>3.085000</td>
      <td>2.320000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.416818</td>
      <td>49842.287197</td>
      <td>2774.399785</td>
      <td>1731.224308</td>
      <td>3.695559</td>
      <td>7.956959</td>
      <td>1.450139</td>
      <td>1.168348</td>
      <td>1.402106</td>
      <td>1.155061</td>
      <td>1.284377</td>
      <td>1.438529</td>
      <td>1.455941</td>
      <td>1.154347</td>
      <td>1.118493</td>
      <td>1.136174</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16.000000</td>
      <td>41775.637023</td>
      <td>-500.000000</td>
      <td>68.817228</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>85832.393634</td>
      <td>204.976456</td>
      <td>420.341127</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.750000</td>
      <td>1.000000</td>
      <td>2.500000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>93868.682835</td>
      <td>328.980863</td>
      <td>1941.855436</td>
      <td>4.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>53.000000</td>
      <td>124572.400926</td>
      <td>597.293077</td>
      <td>2440.774823</td>
      <td>7.000000</td>
      <td>20.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>300.000000</td>
      <td>319704.337941</td>
      <td>50000.000000</td>
      <td>9479.442310</td>
      <td>20.000000</td>
      <td>36.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



由上面的数据发现有什么问题吗？
- `age`：的最大值是300，这不大可能。
- `income`：存在缺失值（816/1000）
- `store_exp`：不应该存在负数、还可能存在离群值，最大消费为50000
- online_exp：看上去没什么问题
- store_trans和online_trans：看上去还比较合理
- Q1~Q10：值的范围都在1~5之内，貌似没问题

那怎么处理这些错误的值呢？这取决于你的实际情况，如果你的样本量很大，不在乎这几个样本，那么就可以删除这些不合理的值。在这里，由于我们只有1000个样本，并且获取这些数据不易，所以得想办法填补这些异常值。我们先把这些值设为缺失状态。


```python
# 将错误的年龄观测设置为缺失值
data['age'].loc[data['age'] > 100] = np.nan
# 将错误的实体店购买设置为缺失值
data['store_exp'].loc[data['store_exp'] < 0] = np.nan
```

    /home/heolis/anaconda3/envs/tensorflow/lib/python3.5/site-packages/pandas/core/indexing.py:194: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._setitem_with_indexer(indexer, value)



```python
# 查看处理后数据的情况
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 19 columns):
    age             999 non-null float64
    gender          1000 non-null object
    income          816 non-null float64
    house           1000 non-null object
    store_exp       999 non-null float64
    online_exp      1000 non-null float64
    store_trans     1000 non-null int64
    online_trans    1000 non-null int64
    Q1              1000 non-null int64
    Q2              1000 non-null int64
    Q3              1000 non-null int64
    Q4              1000 non-null int64
    Q5              1000 non-null int64
    Q6              1000 non-null int64
    Q7              1000 non-null int64
    Q8              1000 non-null int64
    Q9              1000 non-null int64
    Q10             1000 non-null int64
    segment         1000 non-null object
    dtypes: float64(4), int64(12), object(3)
    memory usage: 148.5+ KB


## 缺失值填补
缺失值处理要视情况而定，没有某个方法永远比其他方法好。<br>
在决定处理缺失值值的方法之前，要先了解缺失的原因等关于缺失的辅助信息。
- 缺失是随机发生的吗？如果是，可以用中位数/众数进行填充，也可以使用均值填充。
- 或者说缺失其实是有潜在发生机制的吗？比如年龄大的人在问卷调查中更不愿意透露年龄，这样关于年龄的缺失就不是随机发生的，如果使用均值或者中位数进行填补可能会产生很大偏差。这时需要利用年龄和其他自变量的关系对缺失值进行估计。比如可以基于那些没有缺失值的数据来建模，然后拟合模型预测缺失值。


如果建模的目的是预测，大部分情况下不会很严格地研究缺失机制（缺失机制很明显的时候除外），在缺失机制不太清楚的情况下，可以当成随机缺失进行填补（使用均值中位数或者用K-近邻）

### 中位数或众数填补
对于数值变量我们用中位数进行填补，对于分类变量我们用众数填补


```python
data0 = data.copy()  # 拷贝一份数据，方便对比
data0['age'].fillna(data0['age'].median(), inplace=True)
data0['income'].fillna(data0['income'].median(), inplace=True)
data0['store_exp'].fillna(data0['store_exp'].median(), inplace=True)
```


```python
data0.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 19 columns):
    age             1000 non-null float64
    gender          1000 non-null object
    income          1000 non-null float64
    house           1000 non-null object
    store_exp       1000 non-null float64
    online_exp      1000 non-null float64
    store_trans     1000 non-null int64
    online_trans    1000 non-null int64
    Q1              1000 non-null int64
    Q2              1000 non-null int64
    Q3              1000 non-null int64
    Q4              1000 non-null int64
    Q5              1000 non-null int64
    Q6              1000 non-null int64
    Q7              1000 non-null int64
    Q8              1000 non-null int64
    Q9              1000 non-null int64
    Q10             1000 non-null int64
    segment         1000 non-null object
    dtypes: float64(4), int64(12), object(3)
    memory usage: 148.5+ KB



```python
data0.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
      <th>store_exp</th>
      <th>online_exp</th>
      <th>store_trans</th>
      <th>online_trans</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>Q8</th>
      <th>Q9</th>
      <th>Q10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.576000</td>
      <td>109922.978863</td>
      <td>1357.680319</td>
      <td>2120.181187</td>
      <td>5.350000</td>
      <td>13.546000</td>
      <td>3.101000</td>
      <td>1.823000</td>
      <td>1.992000</td>
      <td>2.763000</td>
      <td>2.945000</td>
      <td>2.448000</td>
      <td>3.434000</td>
      <td>2.396000</td>
      <td>3.085000</td>
      <td>2.320000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.183702</td>
      <td>45660.371065</td>
      <td>2773.967922</td>
      <td>1731.224308</td>
      <td>3.695559</td>
      <td>7.956959</td>
      <td>1.450139</td>
      <td>1.168348</td>
      <td>1.402106</td>
      <td>1.155061</td>
      <td>1.284377</td>
      <td>1.438529</td>
      <td>1.455941</td>
      <td>1.154347</td>
      <td>1.118493</td>
      <td>1.136174</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16.000000</td>
      <td>41775.637023</td>
      <td>155.810945</td>
      <td>68.817228</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>87896.274702</td>
      <td>205.060125</td>
      <td>420.341127</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.750000</td>
      <td>1.000000</td>
      <td>2.500000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>93868.682835</td>
      <td>329.795511</td>
      <td>1941.855436</td>
      <td>4.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>53.000000</td>
      <td>119455.865972</td>
      <td>597.293077</td>
      <td>2440.774823</td>
      <td>7.000000</td>
      <td>20.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>69.000000</td>
      <td>319704.337941</td>
      <td>50000.000000</td>
      <td>9479.442310</td>
      <td>20.000000</td>
      <td>36.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



### K-近邻填补
使用KNN填补缺失值的基本思路是对于含有缺失值的样本，寻找离该样本最近的K个邻居，然后用这些邻居的观测值进行填补。由于这里是根据计算样本点之间的距离来确定邻居的，因此各个变量的标度需要统一，不然尺度大的度量在决定距离上会占主导地位。

这里仅以`income`属性为例。


```python
data1 = data.copy()
# 移除非数值型变量
data1.drop(['gender', 'house', 'segment'], axis=1, inplace=True)

# 用中位数填充age和store_exp
data1['age'].fillna(data1['age'].median(), inplace=True)
data1['store_exp'].fillna(data1['store_exp'].median(), inplace=True)


# 取出income为空的数据作为测试集
test_income = data1.loc[data1['income'].isnull()]
data1.dropna(inplace=True)  # 去除测试集
y_income = data1['income']  # 在预测出点后用于计算平均距离
data1.drop('income', axis=1, inplace=True)
```


```python
# 数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(data1)
data2 = ss.transform(data1)
```


```python
data2 = pd.DataFrame(data2, columns=['age', 'store_exp', 'online_exp', 'store_trans', 'online_trans', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])
data2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>store_exp</th>
      <th>online_exp</th>
      <th>store_trans</th>
      <th>online_trans</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>Q8</th>
      <th>Q9</th>
      <th>Q10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.439428</td>
      <td>-0.290332</td>
      <td>-1.118701</td>
      <td>-0.863047</td>
      <td>-1.571195</td>
      <td>0.686791</td>
      <td>0.154184</td>
      <td>-0.713016</td>
      <td>-0.628257</td>
      <td>-1.677989</td>
      <td>1.165957</td>
      <td>-1.861527</td>
      <td>1.497811</td>
      <td>-1.103135</td>
      <td>1.626789</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.875883</td>
      <td>-0.307717</td>
      <td>-1.232434</td>
      <td>-0.329325</td>
      <td>-1.571195</td>
      <td>0.686791</td>
      <td>-0.695912</td>
      <td>-0.713016</td>
      <td>-0.628257</td>
      <td>-1.677989</td>
      <td>1.165957</td>
      <td>-1.861527</td>
      <td>1.497811</td>
      <td>-2.035942</td>
      <td>1.626789</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.584913</td>
      <td>-0.303363</td>
      <td>-1.132926</td>
      <td>0.471258</td>
      <td>-1.571195</td>
      <td>1.388195</td>
      <td>0.154184</td>
      <td>-0.713016</td>
      <td>-0.628257</td>
      <td>-1.677989</td>
      <td>1.165957</td>
      <td>-1.861527</td>
      <td>1.497811</td>
      <td>-2.035942</td>
      <td>1.626789</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.657656</td>
      <td>-0.351986</td>
      <td>-1.213590</td>
      <td>1.271841</td>
      <td>-1.571195</td>
      <td>1.388195</td>
      <td>0.154184</td>
      <td>-0.713016</td>
      <td>0.219111</td>
      <td>-1.677989</td>
      <td>1.165957</td>
      <td>-1.861527</td>
      <td>1.497811</td>
      <td>-1.103135</td>
      <td>1.626789</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.002973</td>
      <td>-0.341167</td>
      <td>-1.230846</td>
      <td>-0.329325</td>
      <td>-1.315314</td>
      <td>0.686791</td>
      <td>-0.695912</td>
      <td>-0.713016</td>
      <td>0.219111</td>
      <td>-1.677989</td>
      <td>1.165957</td>
      <td>-1.861527</td>
      <td>1.497811</td>
      <td>-1.103135</td>
      <td>1.626789</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(data2)
points = neigh.kneighbors(test_income.drop('income', axis=1), return_distance=False)
```


```python
mean = []
for i in range(len(test_income)):
    mean.append(y_income.iloc[points[i]].mean()) 

test_income['income'] = mean
```

    /home/heolis/anaconda3/envs/tensorflow/lib/python3.5/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """


### Bagging 填充
Bagging是一种集成学习方法，可以用剩余变量训练一个Bagging模型，再用这个模型去预测缺失值，但是它的计算量要大的多。一般来说，如果中位数或者均值填补就能满足建模的需要，使用Bagging的方式填补，就是可以提高一点精度，但是提升的可能会很小，这在样本量很大的使用，就没有太大意义了。

## 中心化和标准化
这是最基本的数据变换。
- 1.中心化是通过将变量的每个观测减去该变量均值，这样中心化后的变量观测值为0。
- 2.标准化是将变量观测除以变量标准差，标准化后的变量标准差为1。

对于一些要对变量进行线性组合的模型，中心化和标准化保证了变量的线性组合是基于组合后的新变量能够解释的 原始变量中的方差。用到基于方差的变量线性组合的模型有主成分分析(PCA)、偏最小二乘分析(PLS)、探索因子分析(EFA)等。

通过参数估计衡量各个自变量和因变量之间关系强度时，必须要对变量观测进行标准化。在仅需要确保参数被“公平”对待时，有时只需要标准化数据而不一定要中心化。这是对数据收缩常用方法。

$$X^*_{i,j} = \frac{X_{i,j} - quantile(X_{i,j}, 0.01)}{quantile(X_{i,j}, 0.99) - quantile(X_{i,j}, 0.01)}$$

这里的$X_{i,j}$代表第个样本的第j个变量观测，$quantile(X_{i,j}, 0.01)$指的是第j个变量所有样本观测组成的向量1%分位数，类似地，$quantile(X_{i,j}, 0.99)$是99%分位数，这里之所以使用99%和1%而非最大最小值，是为了减弱离群点的影响。

## 有偏分布
如果模型要求变量服从一定的对称分布（如正态分布），则需要进行数据变换去除分布中的偏度。<br>
> 偏度是3阶标准化中心[矩](https://zh.wikipedia.org/wiki/%E7%9F%A9_(%E6%95%B8%E5%AD%B8)),是用来衡量分布不对称程度的，该统计两的数学定义如下：
> $$偏度 = \frac {\sum(X_i + \bar x)^3}{(n - 1)v^\frac {2}{3}}$$ 
> $$v=\frac{\sum (x_i = \bar x)^2}{(n-1)}$$
> 数据分布对称时偏度=0，分布左偏时偏度<0，分布右偏时偏度>0，且偏离程度越大，偏度统计量的绝对值越大。

有很多变换有助于去除偏度，如log变换、平方根或者取倒数。Box和Cox（1964）提出了含有一个参数$\lambda$的指数变换族：
$$x^* = \begin{cases}
\frac{x^\lambda - 1}{\lambda}, if(\lambda \ne 0) \\ \log (x), if(\lambda = 0)
\end{cases}$$ 
很容易看出这个变换族群包含了log(x)变换（$\lambda = 0$）、$x^2$变换（$\lambda = 2$）、sqrt(x)变换（$\lambda = 0.5$），以及$fraclx$变换（$\lambda = -1$）等常用的变换。Box-Cox覆盖的面更广，变换指数可能是任意实数。

## 处理离群点
- 1.箱线图和直方图等一些基本可视化可以用来初步检查是否有离群点。
- 2.除了可视化这样直观的方式外，在一定的假设条件下，还有一些统计学的定义离群值的方法。如常用Z分值来判断可能的离群点。

对于某观测变量Y的Z分值定义为：
$$Z_i = \frac{Y_i - \bar Y}{s}$$
其中$\bar Y$和$s$分别为观测列的均值和标准差。直观地理解Z分值就是对观测离均值的距离的度量（多少个标准差单位）。这种方法可能具有误导性，尤其是在样本量小的时候。但Iglewicz 和 Hoaglin 提出了使用修正后的Z分值来判断离群点：
$$M_i = \frac{0.6745(Y_i - \bar Y)}{MAD}$$
其中$MAD$是一系列$|Y_i - \bar Y|$的中位数，称为绝对离差中位数。建议将上面修正后的Z分值大于3.5的点标记为可能的离群点。

离群点的影响取决于你使用的模型，有的模型对离群值很敏感，如线性回归、逻辑回归。有的模型对离群点具有抗性，如基于树的模型、支持向量机模型。此外，**离群点和错误的观测不一样，它是真实的观测，其中包含信息，所以不能随意删除。**

如果你使用的模型对离群点非常敏感，可以使用空间表示变换。该变换将自变量取值映射到高纬的球面上。变换公式如下：
$$X^*_{ij} = \frac{x_{ij}}{\sqrt{\sum_{j=1}^{p}{x^2_{ij}}}}$$
其中$x_{ij}$表示第i个样本对应第j个变量的观测。由公式可见，每个样本都除以它们的平方模。公式的分母其实可以看作是该样本到p维空间0点的欧氏距离，有以下三点需要特别注意：
- 1.在变换前需要对自变量标准化
- 2.于中心化和标准化不用，这个变换操作的对象是所有的自变量。
- 3.如果需要移除变量，这一步必须要在空间表示变换之前，否则会导致一系列问题。

## 共线性 
即相关性，我们可以绘制相关性矩阵图，可视化变量之间的共线性。

两个变量相关性是不是越强越好呢，不然。两个变量高度相关意味着它们含有重复的信息，我们其实不需要将两个变量同时留在模型中。变量高度相关会导致参数估计极为不稳定，所以我们在进行回归之前需要移除一些高度相关的变量，使得模型中变量相关性在一定的范围之内。《应用预测模型》一书中在处理 该问题时的核心思想是**在删除尽可能少的变量的情况下，将变量两两相关性控制在人为设定的一个阈值内。**
> **处理高度相关变量的算法如下：**
- 1.计算自变量的相关系数矩阵；
- 2.找出相关系数绝对值最大的那对自变量（记为自变量A和B）;
- 3.计算 A 和其他自变量相关系数的均值，对 B 也做同样的计算;
- 4.如果 A 的平均相关系数更大，则将 A 移除；如若不然，移除 B;
- 5.重复步骤2到4，直至所有相关系数的绝对值都低于设定的阈值为止。

建议将这个阈值当成一个调优参数，试验不用的值，看那个对应的模型精度更高，建议在0.6~0.8范围内寻找最优的阈值。

## 稀疏变量
除了高度相关的变量以外，我们还需要移除那些观测非常稀疏的变量。一个极端的例子是某变量观测只有一个取值，我们可以将其称为0方差变量。有的可能 只有若干取值，我们称其为近0方差变量。我们需要识别这些变量，然后将其删除。这些变量的存在对如线性回归和逻辑回归这杨2的模型拟合的表现和稳定性会有很大影响，但对决策树模型没有影响。<br>
**通常识别这样的变量有两个法则**
- 不同取值数目和样本量的比值；
- 最常见的取值频数和第二常见的取值频数之间的比值。

具体怎样处理这些变量，需要去试验一下，哪个方法得到的模型精度高就用哪个。

## 编码名义变量
名义变量，又称为虚设变量，是一个指标性质的变量，通常取值0或1。有时你需要将分类变量转化成名义变量，例如在一份问卷中，每个问题有A,B,C,D,E 五个选项，通常应该将其转化为五个名义变量，然后将其中一个选项当作基准选项。

## 连续变量离散化
个人不建议分析师自行地将连续变量离散化，除非客户或相关领域专家给出明确的理由。连续变量的效能通常比区间变量高，你需要权衡将连续变量离散化对可解释性的提升和对模型精确度的损害。注意这里指的是人为主观地将一些连续变量转变为分类变量而非模型检测出的截断点。有一些模型，如分类/回归树和多元自适应回归样条，它们在建模过程中能够估计合适的截断点。但这属于建模，而非数据预处理。
