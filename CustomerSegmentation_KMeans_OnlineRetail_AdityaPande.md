```python
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
```


```python
df = pd.read_excel("Online Retail.xlsx")

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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    541909 non-null  object        
     1   StockCode    541909 non-null  object        
     2   Description  540455 non-null  object        
     3   Quantity     541909 non-null  int64         
     4   InvoiceDate  541909 non-null  datetime64[ns]
     5   UnitPrice    541909 non-null  float64       
     6   CustomerID   406829 non-null  float64       
     7   Country      541909 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 33.1+ MB
    

### From this information we can see that in the Description and CustomerID have the null value


```python
df.isna().sum()
```




    InvoiceNo           0
    StockCode           0
    Description      1454
    Quantity            0
    InvoiceDate         0
    UnitPrice           0
    CustomerID     135080
    Country             0
    dtype: int64



### As we see CustomerID having the 135,080 ,but filling this by random vlaue is not suitable beause without proper CustomerID we cannot assign the customers the transactions
#### so we are going to drop it 


```python
df = df.dropna(subset=["CustomerID"])
```


```python
df.isna().sum()
```




    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 406829 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    406829 non-null  object        
     1   StockCode    406829 non-null  object        
     2   Description  406829 non-null  object        
     3   Quantity     406829 non-null  int64         
     4   InvoiceDate  406829 non-null  datetime64[ns]
     5   UnitPrice    406829 non-null  float64       
     6   CustomerID   406829 non-null  float64       
     7   Country      406829 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 27.9+ MB
    


```python
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 397924 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    397924 non-null  object        
     1   StockCode    397924 non-null  object        
     2   Description  397924 non-null  object        
     3   Quantity     397924 non-null  int64         
     4   InvoiceDate  397924 non-null  datetime64[ns]
     5   UnitPrice    397924 non-null  float64       
     6   CustomerID   397924 non-null  float64       
     7   Country      397924 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 27.3+ MB
    


```python
invalid_qty_df = df[df['Quantity'] <= 0]
invalid_qty_df.head()

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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
invalid_price_df = df[df['UnitPrice'] <= 0]
invalid_price_df.count()

```




    InvoiceNo      40
    StockCode      40
    Description    40
    Quantity       40
    InvoiceDate    40
    UnitPrice      40
    CustomerID     40
    Country        40
    dtype: int64




```python
## here we remove the data that have in the quantity less than 0 or unitprice also same
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
```


```python
invalid_price_df = df[df['UnitPrice'] <= 0]
invalid_price_df.count()
```




    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64




```python
df[['Quantity', 'UnitPrice']].describe()

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
      <th>Quantity</th>
      <th>UnitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>397884.000000</td>
      <td>397884.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.988238</td>
      <td>3.116488</td>
    </tr>
    <tr>
      <th>std</th>
      <td>179.331775</td>
      <td>22.097877</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.001000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>1.950000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.000000</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>8142.750000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.duplicated().sum()
```




    5192




```python
duplicate_rows = df[df.duplicated()]
duplicate_rows.head()
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>517</th>
      <td>536409</td>
      <td>21866</td>
      <td>UNION JACK FLAG LUGGAGE TAG</td>
      <td>1</td>
      <td>2010-12-01 11:45:00</td>
      <td>1.25</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>527</th>
      <td>536409</td>
      <td>22866</td>
      <td>HAND WARMER SCOTTY DOG DESIGN</td>
      <td>1</td>
      <td>2010-12-01 11:45:00</td>
      <td>2.10</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>537</th>
      <td>536409</td>
      <td>22900</td>
      <td>SET 2 TEA TOWELS I LOVE LONDON</td>
      <td>1</td>
      <td>2010-12-01 11:45:00</td>
      <td>2.95</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>539</th>
      <td>536409</td>
      <td>22111</td>
      <td>SCOTTIE DOG HOT WATER BOTTLE</td>
      <td>1</td>
      <td>2010-12-01 11:45:00</td>
      <td>4.95</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>555</th>
      <td>536412</td>
      <td>22327</td>
      <td>ROUND SNACK BOXES SET OF 4 SKULLS</td>
      <td>1</td>
      <td>2010-12-01 11:49:00</td>
      <td>2.95</td>
      <td>17920.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.duplicated(subset=['InvoiceNo', 'StockCode', 'CustomerID']).sum()
```




    10043




```python
dup_rows = df[df.duplicated(subset=['InvoiceNo', 'StockCode', 'CustomerID'], keep=False)]
dup_rows.head()
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>536381</td>
      <td>71270</td>
      <td>PHOTO CLIP LINE</td>
      <td>1</td>
      <td>2010-12-01 09:41:00</td>
      <td>1.25</td>
      <td>15311.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>125</th>
      <td>536381</td>
      <td>71270</td>
      <td>PHOTO CLIP LINE</td>
      <td>3</td>
      <td>2010-12-01 09:41:00</td>
      <td>1.25</td>
      <td>15311.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>483</th>
      <td>536409</td>
      <td>90199C</td>
      <td>5 STRAND GLASS NECKLACE CRYSTAL</td>
      <td>3</td>
      <td>2010-12-01 11:45:00</td>
      <td>6.35</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>485</th>
      <td>536409</td>
      <td>22111</td>
      <td>SCOTTIE DOG HOT WATER BOTTLE</td>
      <td>1</td>
      <td>2010-12-01 11:45:00</td>
      <td>4.95</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>489</th>
      <td>536409</td>
      <td>22866</td>
      <td>HAND WARMER SCOTTY DOG DESIGN</td>
      <td>1</td>
      <td>2010-12-01 11:45:00</td>
      <td>2.10</td>
      <td>17908.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
dup_rows.shape
```




    (19157, 8)




```python
df = df.drop_duplicates(subset=['InvoiceNo', 'StockCode', 'CustomerID'])
```


```python
df.duplicated(subset=['InvoiceNo', 'StockCode', 'CustomerID']).sum()
```




    0




```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>TotalPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>15.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>TotalPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>15.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
  </tbody>
</table>
</div>



## Cleaned Data


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 387841 entries, 0 to 541908
    Data columns (total 9 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    387841 non-null  object        
     1   StockCode    387841 non-null  object        
     2   Description  387841 non-null  object        
     3   Quantity     387841 non-null  int64         
     4   InvoiceDate  387841 non-null  datetime64[ns]
     5   UnitPrice    387841 non-null  float64       
     6   CustomerID   387841 non-null  float64       
     7   Country      387841 non-null  object        
     8   TotalPrice   387841 non-null  float64       
    dtypes: datetime64[ns](1), float64(3), int64(1), object(4)
    memory usage: 29.6+ MB
    


```python
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
```


```python
snapshot_date
```




    Timestamp('2011-12-10 12:50:00')



## Engineered Features


```python
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate':lambda x:(snapshot_date - x.max()).days,
    'InvoiceNo' :'nunique',
    'TotalPrice':'sum'})
rfm.columns=['Recency','Frequency','Monetary']
```


```python
rfm.head()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>326</td>
      <td>1</td>
      <td>77183.60</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>2</td>
      <td>7</td>
      <td>4310.00</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>75</td>
      <td>4</td>
      <td>1595.64</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>19</td>
      <td>1</td>
      <td>1757.55</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>310</td>
      <td>1</td>
      <td>334.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
rfm.shape
```




    (4338, 3)




```python
rfm.isnull().sum()
```




    Recency      0
    Frequency    0
    Monetary     0
    dtype: int64




```python
rfm.describe()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4338.000000</td>
      <td>4338.000000</td>
      <td>4338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>92.536422</td>
      <td>4.272015</td>
      <td>2038.899609</td>
    </tr>
    <tr>
      <th>std</th>
      <td>100.014169</td>
      <td>7.697998</td>
      <td>8976.554606</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>305.145000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51.000000</td>
      <td>2.000000</td>
      <td>658.760000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>142.000000</td>
      <td>5.000000</td>
      <td>1647.905000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>374.000000</td>
      <td>209.000000</td>
      <td>280206.020000</td>
    </tr>
  </tbody>
</table>
</div>



## Normalized Data


```python
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)


```


```python
rfm_scaled = pd.DataFrame(rfm_scaled,columns=rfm.columns,index=rfm.index) 
rfm_scaled.head()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>2.334574</td>
      <td>-0.425097</td>
      <td>8.372184</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>-0.905340</td>
      <td>0.354417</td>
      <td>0.253033</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>-0.175360</td>
      <td>-0.035340</td>
      <td>-0.049385</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>-0.735345</td>
      <td>-0.425097</td>
      <td>-0.031346</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>2.174578</td>
      <td>-0.425097</td>
      <td>-0.189905</td>
    </tr>
  </tbody>
</table>
</div>




```python
rfm_scaled.shape
```




    (4338, 3)




```python
rfm_scaled.isnull().sum()
```




    Recency      0
    Frequency    0
    Monetary     0
    dtype: int64



### Elbow method of cluster


```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```


```python
inertia =[]
k = range(1,11)
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
```


```python
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```


    
![png](output_45_0.png)
    



```python
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

rfm.head()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>Cluster</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>326</td>
      <td>1</td>
      <td>77183.60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>2</td>
      <td>7</td>
      <td>4310.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>75</td>
      <td>4</td>
      <td>1595.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>19</td>
      <td>1</td>
      <td>1757.55</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>310</td>
      <td>1</td>
      <td>334.40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
cluster_summary = rfm.groupby('Cluster').mean()
cluster_summary
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41.166460</td>
      <td>4.741785</td>
      <td>1881.553621</td>
    </tr>
    <tr>
      <th>1</th>
      <td>246.245179</td>
      <td>1.583104</td>
      <td>556.999937</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.857143</td>
      <td>128.714286</td>
      <td>51170.445714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.875000</td>
      <td>38.125000</td>
      <td>113130.529375</td>
    </tr>
  </tbody>
</table>
</div>




```python
rfm['Cluster'].value_counts()
```




    Cluster
    0    3226
    1    1089
    3      16
    2       7
    Name: count, dtype: int64



#### this K-mean clustering result categories into four distict customer segments based on Recency,Recency,Frequency and Monetary Values.These segment represent VIP customer loyal high value customer regular customer and at risk customereach with the different purchase behaviors and business implication


```python
cluster_summary['Monetary'].plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Average Monetary Value')
plt.title('Average Monetary Value by Customer Cluster')
plt.show()
```


    
![png](output_50_0.png)
    


#### The K-Means clustering method has been able to segment customers effectively into four categories based on their purchase activity using Recency, Frequency, and Monetary values. From this analysis, it has been identified that a few customers make a major contribution toward total revenue, whereas a major segment of customers displays moderate as well as low levels of activity. Customer segmentation based on data science helps businesses in gaining knowledge to enhance customer experience, retention, and ultimately revenue maximization in e-commerce.

#### Customer segmentation was done by applying the K-Means clustering method on the normalized RFM features of cleaned online retail transaction data. The Elbow Method was used to select the number of clusters, k = 4, that best described the customer segmentation. These resulting groups can further be defined and classified on the basis of VIP customers, loyal high-value customers, frequent or regular customers, and at-risk customers. Such segments can help in understanding customer behavior and serve as a basis for decision-making for customer loyalty and customization strategies.


```python

```
