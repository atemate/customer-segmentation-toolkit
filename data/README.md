# Datasets

See code that generates the data in the notebooks `../nbs/`.


## Schemas

```bash
$ head -n1 data/**/*.csv  
==> data/01_data_split_offline_online/no_live_data.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country

==> data/01_data_split_offline_online/raw_live_data.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country

==> data/02_data_clean_rows/no_live_data__cleaned.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country,QuantityCanceled,TotalPrice

==> data/03_data_compute_description_keywords/no_live_data__cleaned__keywords.csv <==
heart,vintage,set,bag,box,glass,christmas,design,candle,flower,holder,decoration,metal,retrospot,card,paper,necklac,cake,silver,polkadot,cover,tin,mug,sign,wrap,bracelet,star,pack,bowl,cushion,ivory,tea,garden,home,mini,egg,mirror,gift,cream,bottle,earrings,clock,wood,jar,ring,bird,garland,hook,party,bead,paisley,ribbon,gold,photo,letter,wall,charm,frame,drawer,zinc,easter,water,plate,spot,skull,pencil,cup,children,case,art,enamel,butterfly,tissue,bell,sticker,tray,stand,round,rabbit,magnet,tree,diamante,spaceboy,union,colour,cutlery,storage,pot,book,lunch,chocolate,sweetheart,hair,pantry,feltcraft,light,fairy,birthday,flock,door,bunny,towel,coffee,sweet,trinket,babushka,t-light,notebook,cat,gingham,antique,number,cabinet,baroque,strawberry,wicker,basket,wire,apple,tube,reel,daisy,shell,kit,dog,purse,dinner,hanger,jack,style,pen,retro,drop,hand,jam,tape,woodland,toy,knob,doormat,table,chick,leaf,warmer,kitchen,london,stripe,shape,parasol,ball,travel,wreath,biscuit,regency,doiley,jigsaw,coaster,money,cottage,incense,dish,rack,wooden,piece,cherry,bathroom,girl,fruit,childs,pan,food,images,toadstool,funky,plant,diner,house,lace,medium,hen,wallet,point,0<.<1,1<.<2,2<.<3,3<.<5,5<.<10,.>10

==> data/data.csv <==
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
```
