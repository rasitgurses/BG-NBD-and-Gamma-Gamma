###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################
from datetime import datetime as dt
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions


# 1. flo_data_20.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

df_ = pd.read_csv('flo_data_20k.csv')
df = df_.copy()
df.head(10)


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df,'customer_value_total_ever_offline')
replace_with_thresholds(df,'customer_value_total_ever_online')
# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df['customer_total_spend'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
df['customer_total_number_of_purchases'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
today_date = dt.datetime(2021, 6, 2)
type(today_date)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
df['total_last_order_date'] = df[['last_order_date', 'last_order_date_online']].max(axis=1)

rfm = df.groupby('master_id').agg({'total_last_order_date': [lambda total_last_order_date : (today_date - total_last_order_date.max()).days,###3 DÜZELT
                                                            lambda total_last_order_date: (today_date - total_last_order_date.min()).days],
                                     'master_id' : lambda  master_id: master_id.nunique(),
                                     'customer_total_spend':lambda customer_total_spend:customer_total_spend.sum()})
rfm.columns = ['Recency', 'T', 'Frequency', 'Monetary']
rfm =  rfm.reset_index()
cltv_df = pd.DataFrame()
cltv_df['customer_id'] = rfm['master_id']
cltv_df['recency_cltv_weekly'] = rfm['Recency'] / 7
cltv_df['T_weekly'] = rfm['T'] / 7
cltv_df['frequency'] = rfm['Frequency']
cltv_df['monetary_cltv_avg'] = rfm['Monetary'] / rfm['Frequency']
print(cltv_df)


rfm['Frequency'] = rfm['Frequency'].round().astype(int)



###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df['exp_sales_3_month'] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])


# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df['exp_sales_6_month'] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])
# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.
cltv_df['exp_sales_3_month'].sort_values(ascending=False).head(10)
cltv_df['exp_sales_6_month'].sort_values(ascending=False).head(10)






# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["expected_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary_cltv_avg'])




# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df['cltv'].sort_values(ascending=False).head(20)

cltv_df = cltv_df.reset_index()
###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])





# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cltv_df.groupby('cltv_segment')[{'frequency','recency_cltv_weekly','monetary_cltv_avg'}].agg({'mean'})