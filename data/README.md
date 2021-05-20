# Datasets

See code that generates the data in the notebooks `../nbs/`.


## Schemas

```bash
$ head -n1 data/**/*.csv
==> data/01_data_split_offline_online/no_live_data.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country

==> data/01_data_split_offline_online/raw_live_data.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country

==> data/02_data_clean_rows/no_live_data_cleaned.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country,QuantityCanceled

==> data/data.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
```
