Trong quá trình hoạt động của doanh nghiệp, việc đánh giá nhân viên luôn rất quan trọng. Nhà quản trị luôn cần nắm được đâu là nhân viên tốt, đâu là nhân viên kém, từ đó đưa ra các quyết định về nhân sự hợp lý. Hôm nay mình giới thiệu một task mà mình được giao về đánh giá kết quả hoạt động nhân viên. 


Dữ liệu mình lọc ra gồm kêt quả của mỗi nhân viên trong tháng liên tiếp, biến outcome chính là kết quả của tháng tiếp theo. Các giá trị có dạng int và nằm trong khoảng [1,5] 

Bài toán: chúng ta dự đoán xem kết quả kinh doanh tháng tiếp theo của nhân viên đó sẽ ra sao, trong bài này mình đặc biệt quan tâm đến giá trị outcome bằng 5, chúng ta cần tìm ra các đối tượng này



```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
```

https://github.com/parrt/dtreeviz/blob/master/notebooks/colors.ipynb


```python
df = pd.read_csv('data_q.csv')
df.head()
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
      <th>MONTH</th>
      <th>UNIT_CODE_DESC</th>
      <th>QUINTILE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019/05</td>
      <td>EF00101-EF00101-Nguyễn Văn Quân</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019/05</td>
      <td>FC0036-FC0036 - Lê Ngọc Duy</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019/05</td>
      <td>FC0337-FC0337-Từ Kiện Tuấn</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019/05</td>
      <td>FC0211-FC0211-Nguyễn Hải Đăng</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019/05</td>
      <td>FC0295-FC0295-Nguyễn Duy Linh</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
shapex = 3
shapey = 1
quit = 6
num_month = shapex + shapey

# get month frame
data = pd.DataFrame()
for i in range(num_month, len(df.MONTH.unique())+1):
    months = df.MONTH.unique()[i-num_month:i]
    outmonths = months[-shapey:]
    # select exist 'UNIT_CODE_DESC'
    months_df = [df[df.MONTH == months[i]]['UNIT_CODE_DESC'].tolist() for i in range(shapex-1)] 
    for i, x in enumerate(months_df):
        if i == 0:
            fc_list = list(set(x)) 
        else: 
            fc_list = list(set(fc_list) & set(x))
    for i,month in enumerate(months):
        dt_temp = df[df.MONTH == month][['UNIT_CODE_DESC','QUINTILE']]
        dt_temp.set_index('UNIT_CODE_DESC', inplace=True)
        col_name = f'QUINTILE_{i}'
        dt_temp.columns = [month]
        if i == 0:
            df_temp = dt_temp
        else:
            df_temp = pd.concat([df_temp, dt_temp], axis=1, sort=True)      
    df_temp.dropna(subset=months[:shapex], inplace=True)
    df_temp['outcome'] = df_temp.loc[:, outmonths].mean(axis=1)
    df_temp.drop(outmonths, axis=1, inplace=True)
    df_temp.columns = [f't-{i}' for i in range(shapex-1, -1, -1)]+['outcome']
    data = data.append(df_temp)
data.reset_index(drop=True, inplace=True)
#print(data.shape)
data.dropna(inplace=True)
print(data.head(5))
```

       t-2  t-1  t-0  outcome
    0  5.0  5.0  4.0      5.0
    1  3.0  5.0  2.0      4.0
    2  1.0  1.0  3.0      4.0
    3  5.0  4.0  2.0      5.0
    4  5.0  4.0  1.0      5.0



```python
data.isna().sum()
```




    t-2        0
    t-1        0
    t-0        0
    outcome    0
    dtype: int64




```python
pd.value_counts(data['outcome'])
```




    1.0    2363
    2.0    2172
    3.0    2016
    4.0    1777
    5.0    1516
    Name: outcome, dtype: int64



### Xem xác xuất chuyển trạng thái của dữ liệu:


```python
compare=1
gr0 = data.groupby(['outcome',data.columns[-1-compare]]).agg(len)['t-2']
gr1 = data.groupby(['outcome']).agg(len)['t-2']
gr = gr0/gr1
cr = pd.concat([pd.DataFrame(gr.loc[i]).T for i in range(1,6)], axis=0).values.round(2)
cr_df = pd.DataFrame(cr)
cr_df.columns = range(1,6)
cr_df.index = range(1,6)

fig, ax = plt.subplots(dpi=100)
sns.heatmap(cr, fmt='', annot=True, cmap=sns.color_palette("RdBu_r", 5), ax=ax)
ax.set_xticklabels(range(1,6))
ax.set_yticklabels(range(1,6))
ax.set_ylabel('(t-1)')
ax.set_xlabel('(t)')
plt.title('xác xuất thay đổi trạng thái')
plt.show()
```


![png](output_8_0.png)



```python
def survey(data, compare=1):
    ggr0 = data.groupby(['outcome',data.columns[-1-compare]]).agg(len)['t-2']
    gr1 = data.groupby(['outcome']).agg(len)['t-2']
    gr = gr0/gr1
    #gr = gr.T
    category_names = ['Q1 (t-1)', 'Q2 (t-1)',
                      'Q3 (t-1)', 'Q4 (t-1)', 'Q5 (t-1)']
    results = {
        'Q1': gr.loc[1],
        'Q2': gr.loc[2],
        'Q3': gr.loc[3],
        'Q4': gr.loc[4],
        'Q5': gr.loc[5],
    }
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdBu_r')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5), dpi=100)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.3 else 'black'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(round(c*100, 2))+'%', ha='center', va='center',
                    color=text_color, fontsize=6, fontweight='bold')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    #plt.title(f'Q Last {compare} month')
    ax.set_ylabel('t-1')
    plt.show()
    
    return fig, ax

survey(data,compare=1)
```


![png](output_9_0.png)





    (<Figure size 920x500 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x7fa04df746d0>)



Với bài toán này chúng ta có thể dùng cả hai phương pháp Classification và regression để giải quyết. 

và để dễ trình bày mình sẽ trọn 2 phương pháp :
 - DecisionTreeClassifier
 - DecisionTreeRegressor

Với DecisionTreeClassifier mình sẽ thay đổi biến outcome với các giá trị mục tiêu 5 sẽ đổi thành 1, còn lại là 0 đề đồ thị không quá rối và đạt đươc mục tiêu của bài toán

kết quả cũng khá dễ đọc khi ta vẽ tree ra


```python
from dtreeviz.trees import *
from sklearn.datasets import *
%config InlineBackend.figure_format = 'png'
```


```python
X = data.drop(['outcome'],axis=1).values
y = np.array([0 if yi < 5 else 1 for yi in data.outcome.tolist()])

classifier = tree.DecisionTreeClassifier(max_depth=3)  # limit depth of tree
classifier.fit(X, y)
viz = dtreeviz(classifier, 
               X, 
               y,
               target_name='state',
               orientation ='LR',
               feature_names=list(data.drop(['outcome'],axis=1).columns), 
               class_names=['non_taget', 'taret'],
               #show_node_labels=True,
               #show_just_path=True,
               #precision=1,
              )                
viz.view()
```


```python
data.fillna(6, inplace=True)
X = data.drop(['outcome'],axis=1).values
y = data['outcome'].values
regr = tree.DecisionTreeRegressor(max_depth=3)
regr.fit(X, y)
viz = dtreeviz(regr, 
               X, 
               y,
               target_name='state',
               orientation ='LR',
               feature_names=['t-3','t-2','t-1'], 
               #class_names=['non_Bad', 'Bad'],
               #show_node_labels=True,
               #show_just_path=True,
               #precision=1,
              )                
viz.view() 
```

Sau khi dự đoán được kết quả kinh doanh của nhân viên. Nhà quản trị có thể đưa ra nhiều quyết định kịp thời như khen thưởng các nhân viên xuất xắc hay cắt giảm các nhân viên yếu kém. 

# HMC

http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017


```python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

# create state space and initial state probabilities

states = ['sleeping', 'eating', 'pooping']
pi = [0.35, 0.35, 0.3]
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

```

    sleeping    0.35
    eating      0.35
    pooping     0.30
    Name: states, dtype: float64
    1.0



```python
compare=1
gr0 = data.groupby(['outcome',data.columns[-1-compare]]).agg(len)['t_2']
gr1 = data.groupby(['outcome']).agg(len)['t_2']
gr = gr0/gr1
cr = pd.concat([pd.DataFrame(gr.loc[i]).T for i in range(1,6)], axis=0).values.round(2)
cr
cr_df = pd.DataFrame(cr)
cr_df.columns = range(1,6)
cr_df.index = range(1,6)
cr_df
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.52</td>
      <td>0.24</td>
      <td>0.13</td>
      <td>0.07</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.30</td>
      <td>0.22</td>
      <td>0.15</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.15</td>
      <td>0.25</td>
      <td>0.23</td>
      <td>0.22</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.10</td>
      <td>0.18</td>
      <td>0.26</td>
      <td>0.26</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.07</td>
      <td>0.13</td>
      <td>0.21</td>
      <td>0.27</td>
      <td>0.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pprint import pprint 

# create a function that maps transition probability dataframe 
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        if col in [4,5]:
            for idx in Q.index:
                edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(cr_df)
print(edges_wts)
```

    {(1, 4): 0.07, (2, 4): 0.15, (3, 4): 0.22, (4, 4): 0.26, (5, 4): 0.27, (1, 5): 0.04, (2, 5): 0.08, (3, 5): 0.15, (4, 5): 0.2, (5, 5): 0.31}



```python
cr_df
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.52</td>
      <td>0.24</td>
      <td>0.13</td>
      <td>0.07</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.25</td>
      <td>0.30</td>
      <td>0.22</td>
      <td>0.15</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.15</td>
      <td>0.25</td>
      <td>0.23</td>
      <td>0.22</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.10</td>
      <td>0.18</td>
      <td>0.26</td>
      <td>0.26</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.07</td>
      <td>0.13</td>
      <td>0.21</td>
      <td>0.27</td>
      <td>0.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(state_space)
#print(f'Nodes:\n{G.nodes()}\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
#print(f'Edges:')
#pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
f, ax = plt.subplots(figsize=(12,6))
nx.draw_networkx(G, pos, ax=ax)

# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}

nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels, rotate=True, font_size=12)
#nx.drawing.nx_pydot.write_dot(G, 'pet_dog_markov.dot')
```




    {(1, 4): Text(226.5, 148.5, '0.07'),
     (1, 5): Text(155.0, 105.0, '0.04'),
     (4, 4): Text(282.0, 105.0, '0.26'),
     (4, 5): Text(210.5, 61.5, '0.2'),
     (2, 4): Text(268.0, 148.5, '0.15'),
     (2, 5): Text(196.5, 105.0, '0.08'),
     (3, 4): Text(307.0, 148.5, '0.22'),
     (3, 5): Text(235.5, 105.0, '0.15'),
     (5, 4): Text(210.5, 61.5, '0.27'),
     (5, 5): Text(139.0, 18.0, '0.31')}




![png](output_20_1.png)



```python
import pandas as pd
import pandas_datareader.data as web
import sklearn.mixture as mix

import numpy as np
import scipy.stats as scs

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
%matplotlib inline

import seaborn as sns
import missingno as msno
from tqdm import tqdm
p=print

```


```python

```

    Collecting missingno
      Downloading missingno-0.4.2-py3-none-any.whl (9.7 kB)
    Requirement already satisfied: seaborn in /home/max/anaconda3/lib/python3.7/site-packages (from missingno) (0.10.0)
    Requirement already satisfied: numpy in /home/max/anaconda3/lib/python3.7/site-packages (from missingno) (1.18.1)
    Requirement already satisfied: matplotlib in /home/max/anaconda3/lib/python3.7/site-packages (from missingno) (3.1.3)
    Requirement already satisfied: scipy in /home/max/anaconda3/lib/python3.7/site-packages (from missingno) (1.4.1)
    Requirement already satisfied: pandas>=0.22.0 in /home/max/anaconda3/lib/python3.7/site-packages (from seaborn->missingno) (1.0.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /home/max/anaconda3/lib/python3.7/site-packages (from matplotlib->missingno) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /home/max/anaconda3/lib/python3.7/site-packages (from matplotlib->missingno) (2.4.6)
    Requirement already satisfied: cycler>=0.10 in /home/max/anaconda3/lib/python3.7/site-packages (from matplotlib->missingno) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /home/max/anaconda3/lib/python3.7/site-packages (from matplotlib->missingno) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /home/max/anaconda3/lib/python3.7/site-packages (from pandas>=0.22.0->seaborn->missingno) (2019.3)
    Requirement already satisfied: setuptools in /home/max/anaconda3/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->missingno) (45.2.0.post20200210)
    Requirement already satisfied: six in /home/max/anaconda3/lib/python3.7/site-packages (from cycler>=0.10->matplotlib->missingno) (1.14.0)
    Installing collected packages: missingno
    Successfully installed missingno-0.4.2

