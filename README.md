# Featurologists
> Engineering Labs #2: Feature Store for ML


## Install

`pip install -U git+https://github.com/artemlops/featurologists.git@master`

## Usage

### 01. Load and split dataset

```
from featurologists.data.load_split import load_csv, split_offline_online
```

```
df = load_csv('../data/data.csv')
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
      <td>17850</td>
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
      <td>17850</td>
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
      <td>17850</td>
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
      <td>17850</td>
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
      <td>17850</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>



```
import datetime

DATE_SPLIT = datetime.date(2011,10,1)
df_offline, df_online = split_offline_online(df, DATE_SPLIT)

display(df_offline.shape)
display(df_offline['InvoiceDate'].min())
display(df_offline['InvoiceDate'].max())

display(df_online.shape)
display(df_online['InvoiceDate'].min())
display(df_online['InvoiceDate'].max())
```


    (370931, 8)



    Timestamp('2010-12-01 08:26:00')



    Timestamp('2011-09-30 17:22:00')



    (170978, 8)



    Timestamp('2011-10-02 10:32:00')



    Timestamp('2011-12-09 12:50:00')


```
NB_NAME = '01_data_split_offline_online'
OUT_DATA_PATH = f'../data/{NB_NAME}'
!mkdir -p {OUT_DATA_PATH}

df_offline.to_csv(f'{OUT_DATA_PATH}/no_live_data.csv', index=False)
df_online.to_csv(f'{OUT_DATA_PATH}/raw_live_data.csv', index=False)
del df, df_offline, df_online;
```

### 02. Clean dataset rows

```
from featurologists.data.load_split import load_csv
from featurologists.data.clean_rows import clean_rows
```

```
df = load_csv('../data/01_data_split_offline_online/no_live_data.csv')
df_cleaned = clean_rows(df)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-35-617a1f2e66fb> in <module>
          1 df = load_csv('../data/01_data_split_offline_online/no_live_data.csv')
    ----> 2 df_cleaned = clean_rows(df)
    

    /plain/github/mine/featurologists/featurologists/data/clean_rows.py in clean_rows(df)
         78     df = clean_drop_na(df)
         79     df = clean_drop_duplicates(df)
    ---> 80     df = clean_remove_bad_order_cancellations(df)
         81     return df


    /plain/github/mine/featurologists/featurologists/data/clean_rows.py in clean_remove_bad_order_cancellations(df)
         37         df_test = df[(df['CustomerID'] == col['CustomerID']) &
         38                      (df['StockCode']  == col['StockCode']) &
    ---> 39                      (df['InvoiceDate'] < col['InvoiceDate']) &
         40                      (df['Quantity']   > 0)].copy()
         41 


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/ops/common.py in new_method(self, other)
         63         other = item_from_zerodim(other)
         64 
    ---> 65         return method(self, other)
         66 
         67     return new_method


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/arraylike.py in __lt__(self, other)
         35     @unpack_zerodim_and_defer("__lt__")
         36     def __lt__(self, other):
    ---> 37         return self._cmp_method(other, operator.lt)
         38 
         39     @unpack_zerodim_and_defer("__le__")


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/series.py in _cmp_method(self, other, op)
       4976         rvalues = extract_array(other, extract_numpy=True)
       4977 
    -> 4978         res_values = ops.comparison_op(lvalues, rvalues, op)
       4979 
       4980         return self._construct_result(res_values, name=res_name)


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/ops/array_ops.py in comparison_op(left, right, op)
        227     if should_extension_dispatch(lvalues, rvalues):
        228         # Call the method on lvalues
    --> 229         res_values = op(lvalues, rvalues)
        230 
        231     elif is_scalar(rvalues) and isna(rvalues):


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/ops/common.py in new_method(self, other)
         63         other = item_from_zerodim(other)
         64 
    ---> 65         return method(self, other)
         66 
         67     return new_method


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/arraylike.py in __lt__(self, other)
         35     @unpack_zerodim_and_defer("__lt__")
         36     def __lt__(self, other):
    ---> 37         return self._cmp_method(other, operator.lt)
         38 
         39     @unpack_zerodim_and_defer("__le__")


    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/pandas/core/arrays/datetimelike.py in _cmp_method(self, other, op)
        950 
        951         o_mask = isna(other)
    --> 952         mask = self._isnan | o_mask
        953         if mask.any():
        954             nat_result = op is operator.ne


    KeyboardInterrupt: 


```
NB_NAME = '02_data_clean_rows'
OUT_DATA_PATH = f'../data/{NB_NAME}'
!mkdir -p {OUT_DATA_PATH}

df_cleaned.to_csv(f'{OUT_DATA_PATH}/no_live_data__cleaned.csv', index=False)
del df, df_cleaned;
```

### 03. Analyse keywords in product descriptions

```
from featurologists.data.load_split import load_csv
from featurologists.data.analyse_keywords import build_product_list, build_keywords_matrix
```

```
df_cleaned = load_csv('../data/02_data_clean_rows/no_live_data__cleaned.csv')
list_products = build_product_list(df)
list_products[:5]
```

```
THRESHOLD = [0, 1, 2, 3, 5, 10]
X = build_keywords_matrix(df_cleaned, list_products, THRESHOLD)
X.head()
```

```
NB_NAME = '03_data_compute_description_keywords'
OUT_DATA_PATH = f'../data/{NB_NAME}'
!mkdir -p {OUT_DATA_PATH}

X.to_csv(f'{OUT_DATA_PATH}/no_live_data__cleaned__keywords.csv', index=False)
del df_cleaned, list_products, X;
```
