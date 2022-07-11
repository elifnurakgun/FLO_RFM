import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

df_ = pd.read_csv("datasets/flo_data_20K.csv")
df = df_.copy()

df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()


date_columns = df.columns[df.columns.str.contains("date")] # Fancy Index
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()



# Alışveriş kanallarındaki / kırılımındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakmak
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total": "sum"})

# En fazla kazancı getiren ilk 10 müşteri
df.sort_values("customer_value_total", ascending=False)[:10]
df.sort_values("customer_value_total", ascending=False).head(10)

# En fazla siparişi veren ilk 10 müşteri
df.sort_values("order_num_total", ascending=False)[:10]
df.sort_values("order_num_total", ascending=False).head(10)


# Veri ön hazırlık sürecini fonksiyonlaştırmak
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe


df = data_prep(df)

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi
# 2021-05-30
df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)


# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
# rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()
rfm.info()

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()

# recency_score ve frequency_score ve monetary_score'u tek bir değişken olarak ifade edilmesi ve RFM_SCORE olarak kaydedilmesi
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

rfm.head()

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

# Segmentlerin recency, frequnecy ve monetary ortalamaları
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                          recency       frequency       monetary
#                        mean count      mean count     mean count
# segment
# about_to_sleep       113.79  1629      2.40  1629   359.01  1629
# at_Risk              241.61  3131      4.47  3131   646.61  3131
# cant_loose           235.44  1200     10.70  1200  1474.47  1200
# champions             17.11  1932      8.93  1932  1406.63  1932
# hibernating          247.95  3604      2.39  3604   366.27  3604
# loyal_customers       82.59  3361      8.37  3361  1216.82  3361
# need_attention       113.83   823      3.73   823   562.14   823
# new_customers         17.92   680      2.00   680   339.96   680
# potential_loyalists   37.16  2938      3.30  2938   533.18  2938
# promising             58.92   647      2.00   647   335.67   647


# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin şampiyon ve sadık, ortalama 250 TL üzeri ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["customer_value_total"] / df["order_num_total"] > 250) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

# Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)
cust_ids.shape




  


