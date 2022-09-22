# Importing Libraries


```python
# All library has been used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import pylab as pylab
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
```

# Loading Data


```python
df=pd.read_csv("data_raw.csv",index_col=0)
```


```python
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
      <th>sku</th>
      <th>weekly_sales</th>
      <th>feat_main_page</th>
      <th>color</th>
      <th>price</th>
      <th>vendor</th>
      <th>functionality</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10/31/2016</th>
      <td>1</td>
      <td>135</td>
      <td>True</td>
      <td>black</td>
      <td>10.16</td>
      <td>6</td>
      <td>06.Mobile phone accessories</td>
    </tr>
    <tr>
      <th>11/7/2016</th>
      <td>1</td>
      <td>102</td>
      <td>True</td>
      <td>black</td>
      <td>9.86</td>
      <td>6</td>
      <td>06.Mobile phone accessories</td>
    </tr>
    <tr>
      <th>11/14/2016</th>
      <td>1</td>
      <td>110</td>
      <td>True</td>
      <td>black</td>
      <td>10.24</td>
      <td>6</td>
      <td>06.Mobile phone accessories</td>
    </tr>
    <tr>
      <th>11/21/2016</th>
      <td>1</td>
      <td>127</td>
      <td>True</td>
      <td>black</td>
      <td>8.27</td>
      <td>6</td>
      <td>06.Mobile phone accessories</td>
    </tr>
    <tr>
      <th>11/28/2016</th>
      <td>1</td>
      <td>84</td>
      <td>True</td>
      <td>black</td>
      <td>8.83</td>
      <td>6</td>
      <td>06.Mobile phone accessories</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (4400, 7)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4400 entries, 10/31/2016 to 9/24/2018
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   sku             4400 non-null   int64  
     1   weekly_sales    4400 non-null   int64  
     2   feat_main_page  4400 non-null   bool   
     3   color           4390 non-null   object 
     4   price           4400 non-null   float64
     5   vendor          4400 non-null   int64  
     6   functionality   4400 non-null   object 
    dtypes: bool(1), float64(1), int64(3), object(2)
    memory usage: 244.9+ KB
    


```python
df.describe()
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
      <th>sku</th>
      <th>weekly_sales</th>
      <th>price</th>
      <th>vendor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4400.000000</td>
      <td>4400.000000</td>
      <td>4400.000000</td>
      <td>4400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.500000</td>
      <td>83.054773</td>
      <td>44.432709</td>
      <td>6.909091</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.699868</td>
      <td>288.000205</td>
      <td>42.500295</td>
      <td>2.503175</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.390000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.750000</td>
      <td>11.000000</td>
      <td>15.680000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.500000</td>
      <td>25.000000</td>
      <td>27.550000</td>
      <td>6.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.250000</td>
      <td>70.000000</td>
      <td>54.990000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>44.000000</td>
      <td>7512.000000</td>
      <td>227.720000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Data Preprocessing


```python
df.isnull().sum()
```




    sku                0
    weekly_sales       0
    feat_main_page     0
    color             10
    price              0
    vendor             0
    functionality      0
    dtype: int64



#### Observed that there are 10 na values ​​in color attribute, so decided to use dropna method


```python
df.replace("",inplace=True)
df.dropna(subset=["color"],inplace=True)
```


```python
df.isnull().sum()
```




    sku               0
    weekly_sales      0
    feat_main_page    0
    color             0
    price             0
    vendor            0
    functionality     0
    dtype: int64




```python
df.shape
```




    (4390, 7)



### Take a look some features


```python
plt.bar(df['color'],df['weekly_sales'],data=df,color='blue')
plt.title('Weekly sales by color')
plt.xlabel('Color')
plt.ylabel('Weekly sales')
plt.show()
```


    
![png](output_15_0.png)
    



```python
plt.scatter(df['sku'],df['weekly_sales'])
plt.title('Weekly sales by sku')
plt.xlabel('SKU')
plt.ylabel('weekly sales')
plt.show()
```


    
![png](output_16_0.png)
    



```python
dataframe =df.copy()
dataframe.groupby(['feat_main_page']).sum().plot(
    kind='pie', y='weekly_sales', autopct='%1.0f%%',startangle=60,title='Percentage of feat_main_page feature',ylabel='')
```




    <AxesSubplot:title={'center':'Percentage of feat_main_page feature'}>




    
![png](output_17_1.png)
    


## Converting Categorical Values


```python
# Making copy to avoid changing original data
dummy=df.copy()
cats=['color','functionality','feat_main_page']
label_encoder=LabelEncoder()
for i in cats:
    dummy[i]=label_encoder.fit_transform(dummy[i])
dummy.head()
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
      <th>sku</th>
      <th>weekly_sales</th>
      <th>feat_main_page</th>
      <th>color</th>
      <th>price</th>
      <th>vendor</th>
      <th>functionality</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10/31/2016</th>
      <td>1</td>
      <td>135</td>
      <td>1</td>
      <td>0</td>
      <td>10.16</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11/7/2016</th>
      <td>1</td>
      <td>102</td>
      <td>1</td>
      <td>0</td>
      <td>9.86</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11/14/2016</th>
      <td>1</td>
      <td>110</td>
      <td>1</td>
      <td>0</td>
      <td>10.24</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11/21/2016</th>
      <td>1</td>
      <td>127</td>
      <td>1</td>
      <td>0</td>
      <td>8.27</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11/28/2016</th>
      <td>1</td>
      <td>84</td>
      <td>1</td>
      <td>0</td>
      <td>8.83</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# correlation matrix
figure,axs=plt.subplots(figsize=(12,10))
sns.heatmap(dummy.corr(),annot=True)
```




    <AxesSubplot:>




    
![png](output_20_1.png)
    


**Points to notice:**
- "sku", "feat_main_page", "color" and "carat" show a high correlation to the "weekly_sales" column
- The orthes show low correlation. We could consider dropping but let's keep it

## Model Building
**Steps** involved in Model Building
- Setting up features and target
- Build a pipeline of standard scalar and model for five different regressors.
- Fit all models on training data
- Get mean of cross-validation on the training set for all the models for negative root mean square error
- Pick the model with the best cross-validation score
- Fit the best model on the training set and get


```python
# Assigning the features as X and the target as y
X = dummy.drop(['weekly_sales'],axis=1)
y = dummy['weekly_sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=7)
```


```python
# Building pipelines of standard scaler and model for varios regressor
pipeline_lr=Pipeline([("scalar1",StandardScaler()),("lr_classifier",LinearRegression())])
pipeline_dt=Pipeline([("scalar2",StandardScaler()),("dt_classifier",DecisionTreeRegressor())])
pipeline_rf=Pipeline([("scalar3",StandardScaler()),("rf_classifier",RandomForestRegressor())])
pipeline_kn=Pipeline([("scalar4",StandardScaler()),("kn_classifier",KNeighborsRegressor())])
pipeline_xgb=Pipeline([("scalar5",StandardScaler()),("xgb_classifier",XGBRegressor())])

# List of all the pipelines
pipelines=[pipeline_lr,pipeline_dt,pipeline_rf,pipeline_kn,pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipe_dict={0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train,y_train)
```


```python
cv_result_rms=[]
for i, model in enumerate(pipelines):
    cv_score=cross_val_score(model,X_train,y_train,scoring="neg_root_mean_squared_error",cv=10)
    cv_result_rms.append(cv_score)
    print("%s: %f"%(pipe_dict[i],cv_score.mean()))
```

    LinearRegression: -271.737516
    DecisionTree: -256.348723
    RandomForest: -196.538705
    KNeighbors: -184.760909
    XGBRegressor: -199.548701
    

**Testing the Model with the best score on the test set**

In the above scores, KNN appears to be the model with the best scoring on negative root mean squared error. Let's test this model on a test set and evaluate it with different parameters


```python
# Model prediction on test data
y_pred=pipeline_kn.predict(X_test)
```


```python
# Model evaluation
print("R^2:",metrics.r2_score(y_test,y_pred))
print("Adjusted R^2:",1-(1-metrics.r2_score(y_test,y_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
```

    R^2: 0.5701590737980159
    Adjusted R^2: 0.5677951456979133
    MAE: 41.784335154826955
    MSE: 24726.89683060109
    RMSE: 157.24788338989205
    


```python
pd.DataFrame({'y':y_test, 'y_pred_kn':y_pred})
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
      <th>y</th>
      <th>y_pred_kn</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8/20/2018</th>
      <td>20</td>
      <td>22.6</td>
    </tr>
    <tr>
      <th>12/25/2017</th>
      <td>33</td>
      <td>35.8</td>
    </tr>
    <tr>
      <th>10/30/2017</th>
      <td>89</td>
      <td>121.8</td>
    </tr>
    <tr>
      <th>4/23/2018</th>
      <td>74</td>
      <td>75.2</td>
    </tr>
    <tr>
      <th>6/11/2018</th>
      <td>63</td>
      <td>82.6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3/12/2018</th>
      <td>73</td>
      <td>23.6</td>
    </tr>
    <tr>
      <th>10/31/2016</th>
      <td>46</td>
      <td>123.0</td>
    </tr>
    <tr>
      <th>9/10/2018</th>
      <td>24</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>1/15/2018</th>
      <td>71</td>
      <td>89.6</td>
    </tr>
    <tr>
      <th>8/13/2018</th>
      <td>5</td>
      <td>4.4</td>
    </tr>
  </tbody>
</table>
<p>1098 rows × 2 columns</p>
</div>




```python
y_preds=pd.Series(y_pred))
plt.plot(y_test.head(10),label='Actual Sales (Testing)')
plt.plot(y_preds.head(10),label='Predicted Sales')
plt.xticks(rotation=30)
plt.legend(loc='upper right')
plt.title('Weekly sales predicting')
plt.ylabel('Sales')
plt.show()
```


    
![png](output_30_0.png)
    


**Testing other models**


```python
y_pred1=pipeline_lr.predict(X_test)
y_pred2=pipeline_dt.predict(X_test)
y_pred3=pipeline_rf.predict(X_test)
y_pred4=pipeline_xgb.predict(X_test)
```


```python
pd.DataFrame({'y':y_test, 'y_preds_lr':y_pred1, 'y_preds_dt':y_pred2, 'y_pred_rf':y_pred3, 'y_pred_xgb':y_pred4})
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
      <th>y</th>
      <th>y_preds_lr</th>
      <th>y_preds_dt</th>
      <th>y_pred_rf</th>
      <th>y_pred_xgb</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8/20/2018</th>
      <td>20</td>
      <td>37.672429</td>
      <td>24.000000</td>
      <td>33.293333</td>
      <td>23.476191</td>
    </tr>
    <tr>
      <th>12/25/2017</th>
      <td>33</td>
      <td>101.372166</td>
      <td>33.000000</td>
      <td>34.145857</td>
      <td>39.714310</td>
    </tr>
    <tr>
      <th>10/30/2017</th>
      <td>89</td>
      <td>99.773412</td>
      <td>90.000000</td>
      <td>110.800000</td>
      <td>123.906006</td>
    </tr>
    <tr>
      <th>4/23/2018</th>
      <td>74</td>
      <td>65.272455</td>
      <td>79.000000</td>
      <td>70.960000</td>
      <td>47.384869</td>
    </tr>
    <tr>
      <th>6/11/2018</th>
      <td>63</td>
      <td>142.355789</td>
      <td>71.000000</td>
      <td>80.245000</td>
      <td>88.312614</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3/12/2018</th>
      <td>73</td>
      <td>93.529993</td>
      <td>142.000000</td>
      <td>72.018333</td>
      <td>35.488598</td>
    </tr>
    <tr>
      <th>10/31/2016</th>
      <td>46</td>
      <td>176.925865</td>
      <td>237.000000</td>
      <td>163.080000</td>
      <td>213.307831</td>
    </tr>
    <tr>
      <th>9/10/2018</th>
      <td>24</td>
      <td>76.332554</td>
      <td>34.000000</td>
      <td>34.285000</td>
      <td>38.932377</td>
    </tr>
    <tr>
      <th>1/15/2018</th>
      <td>71</td>
      <td>69.461707</td>
      <td>89.000000</td>
      <td>87.970000</td>
      <td>81.578812</td>
    </tr>
    <tr>
      <th>8/13/2018</th>
      <td>5</td>
      <td>24.885877</td>
      <td>4.333333</td>
      <td>4.241714</td>
      <td>6.693056</td>
    </tr>
  </tbody>
</table>
<p>1098 rows × 5 columns</p>
</div>




```python
y_pred1s=pd.Series(y_pred1)
y_pred2s=pd.Series(y_pred2)
y_pred3s=pd.Series(y_pred3)
y_pred4s=pd.Series(y_pred4)
plt.plot(y_test.head(10),label='Actual Sales (Testing)')
plt.plot(y_pred1s.head(10),label='Linear Regression',alpha=0.3,color="skyblue")
plt.plot(y_pred2s.head(10),label='Decision Tree',linestyle='dashed')
plt.plot(y_pred3s.head(10),label='Forest',linestyle='-.')
plt.plot(y_pred4s.head(10),label='XGB',linestyle=':')
plt.xticks(rotation=30)
plt.legend(loc='lower left')
plt.title('Weekly sales predicting')
plt.ylabel('Sales')
plt.show()
```


    
![png](output_34_0.png)
    

