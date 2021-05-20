# Featurologists
> Engineering Labs #2: Feature Store for ML


## Install

`pip install -U git+https://github.com/artemlops/featurologists.git@master`

## Usage

### 01. Load and split dataset

```python
from featurologists.data.load_split import load_csv, split_offline_online
```

```python
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



```python
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


```python
NB_NAME = '01_data_split_offline_online'
OUT_DATA_PATH = f'../data/{NB_NAME}'
!mkdir -p {OUT_DATA_PATH}

df_offline.to_csv(f'{OUT_DATA_PATH}/no_live_data.csv', index=False)
df_online.to_csv(f'{OUT_DATA_PATH}/raw_live_data.csv', index=False)
del df, df_offline, df_online;
```

### 02. Clean dataset rows

```python
from featurologists.data.load_split import load_csv
from featurologists.data.clean_rows import clean_data_rows
```

```python
df = load_csv('../data/01_data_split_offline_online/no_live_data.csv')
df_cleaned = clean_data_rows(df)
```

```python
NB_NAME = '02_data_clean_rows'
OUT_DATA_PATH = f'../data/{NB_NAME}'
!mkdir -p {OUT_DATA_PATH}

df_cleaned.to_csv(f'{OUT_DATA_PATH}/no_live_data__cleaned.csv', index=False)
del df, df_cleaned;
```

### 03. Analyse keywords in product descriptions

```python
from featurologists.data.load_split import load_csv
from featurologists.data.analyse_keywords import build_product_list, build_keywords_matrix
```

```python
df_cleaned = load_csv('../data/02_data_clean_rows/no_live_data__cleaned.csv')
list_products = build_product_list(df_cleaned)
list_products[:5]
```

    /plain/github/mine/featurologists/venv/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3361: DtypeWarning: Columns (0) have mixed types.Specify dtype option on import or set low_memory=False.
      if (await self.run_code(code, result,  async_=asy)):





    [['heart', 251], ['vintage', 195], ['set', 194], ['bag', 158], ['box', 147]]



```python
THRESHOLD = [0, 1, 2, 3, 5, 10]
X = build_keywords_matrix(df_cleaned, list_products, THRESHOLD)
X.head()
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
      <th>heart</th>
      <th>vintage</th>
      <th>set</th>
      <th>bag</th>
      <th>box</th>
      <th>glass</th>
      <th>christmas</th>
      <th>design</th>
      <th>candle</th>
      <th>flower</th>
      <th>...</th>
      <th>medium</th>
      <th>hen</th>
      <th>wallet</th>
      <th>point</th>
      <th>0&lt;.&lt;1</th>
      <th>1&lt;.&lt;2</th>
      <th>2&lt;.&lt;3</th>
      <th>3&lt;.&lt;5</th>
      <th>5&lt;.&lt;10</th>
      <th>.&gt;10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 188 columns</p>
</div>



```python
NB_NAME = '03_data_compute_description_keywords'
OUT_DATA_PATH = f'../data/{NB_NAME}'
!mkdir -p {OUT_DATA_PATH}

X.to_csv(f'{OUT_DATA_PATH}/no_live_data__cleaned__keywords.csv', index=False)
del df_cleaned, list_products, X;
```
