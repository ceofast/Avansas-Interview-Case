import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from datetime import datetime, timedelta
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format',lambda x: '%.5f' % x)

df = pd.read_excel(r"D:\avansas_case\DS_Öğrenme_Verisi.xlsx")
df_test = pd.read_excel(r"D:\avansas_case\DS_Uygulama_Verisi.xlsx")
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df['SON SİPARİŞ TARİHİ'] = pd.to_datetime(df['SON SİPARİŞ TARİHİ'])

df['SON SİPARİŞ TARİHİ'].max()

today = dt.datetime(2023, 1, 24)

three_months_ago = today - timedelta(days=90)

last_3_months = df[df['SON SİPARİŞ TARİHİ'] >= three_months_ago]

last_3_months_orders = last_3_months.groupby('MÜŞTERI NO').agg({'TOPLAM SİPARİŞ ADEDİ': 'sum', 'TOPLAM SİPARİŞ HACMİ': 'sum'}).reset_index()

last_3_months_orders.columns = ['MÜŞTERI NO', 'SON 3 AY SİPARİŞ ADEDİ', 'SON 3 AY SİPARİŞ HACMİ']

df.drop(columns=['SON 3 AY SİPARİŞ ADEDİ', 'SON 3 AY SİPARİŞ HACMİ'], inplace= True)

merged_data = pd.merge(df, last_3_months_orders, on='MÜŞTERI NO', how='left')

merged_data.head()

# Ortalama sipariş adedi ve hacmi hesaplanıyor
mean_total_orders = merged_data['TOPLAM SİPARİŞ ADEDİ'].mean()
mean_last_6_months_orders = merged_data['SON 6 AY SİPARİŞ ADEDİ'].mean()
mean_total_order_amount = merged_data['TOPLAM SİPARİŞ HACMİ'].mean()
mean_last_6_months_order_amount = merged_data['SON 6 AY SİPARİŞ HACMİ'].mean()
mean_last_3_months_orders = merged_data['SON 3 AY SİPARİŞ ADEDİ'].mean()
mean_last_3_months_order_amount = merged_data['SON 3 AY SİPARİŞ HACMİ'].mean()

print("Ortalama Toplam Sipariş Adedi: ", mean_total_orders)
print("Ortalama Son 6 Aylık Sipariş Adedi: ", mean_last_6_months_orders)
print("Ortalama Toplam Sipariş Hacmi:", mean_total_order_amount)
print("Ortalama Son 6 Ay Sipariş Hacmi:", mean_last_6_months_order_amount)
print("Ortalama Son 3 Ay Sipariş Adedi:", mean_last_3_months_orders)
print("Ortalama Son 3 Ay Sipariş Hacmi:", mean_last_3_months_order_amount)

aktif = []
for i in range(len(merged_data)):
    if merged_data.loc[i, "SON 6 AY SİPARİŞ ADEDİ"] > 10 and merged_data.loc[i, "SON 6 AY SİPARİŞ HACMİ"] > 500:
        aktif.append(1)
    else:
        aktif.append(0)

merged_data["AKTİF/PASİF"] = np.array(aktif)

# Aktif değişkeni hesaplanıyor
# merged_data['AKTİF'] = (merged_data['SON 6 AY SİPARİŞ ADEDİ'] > mean_last_6_months_orders).astype(int) & \
#                       (merged_data['TOPLAM SİPARİŞ HACMİ'] > mean_total_order_amount).astype(int) & \
#                       (merged_data['SON 6 AY SİPARİŞ HACMİ'] > mean_last_6_months_order_amount).astype(int) & \
#                       (merged_data['TOPLAM SİPARİŞ ADEDİ'] > mean_total_orders).astype(int) & \
#                       (merged_data['SON 3 AY SİPARİŞ ADEDİ'] > mean_last_3_months_orders).astype(int) & \
#                       (merged_data['SON 3 AY SİPARİŞ HACMİ'] > mean_last_3_months_order_amount).astype(int)

merged_data = merged_data.dropna(subset=['SON SİPARİŞ TARİHİ'])
merged_data.head()
merged_data['AKTİF/PASİF'].value_counts()

def grab_col_names(dataframe, cat_th=25, car_th=50):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(merged_data)

df_test['SON SİPARİŞ TARİHİ'] = pd.to_datetime(df_test['SON SİPARİŞ TARİHİ'])

df_test['SON SİPARİŞ TARİHİ'].max()

today = dt.datetime(2023, 1, 24)
three_months_ago = today - timedelta(days=90)

last_3_months_test = df_test[df_test['SON SİPARİŞ TARİHİ'] >= three_months_ago]

last_3_months_orders_test = last_3_months_test.groupby('MÜŞTERI NO').agg({'TOPLAM SİPARİŞ ADEDİ': 'sum', 'TOPLAM SİPARİŞ HACMİ': 'sum'}).reset_index()

last_3_months_orders_test.columns = ['MÜŞTERI NO', 'SON 3 AY SİPARİŞ ADEDİ', 'SON 3 AY SİPARİŞ HACMİ']

df_test.drop(columns=['SON 3 AY SİPARİŞ ADEDİ', 'SON 3 AY SİPARİŞ HACMİ'], inplace= True)

merged_data_test = pd.merge(df_test, last_3_months_orders_test, on='MÜŞTERI NO', how='left')

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(merged_data_test)

merged_data["LOKASYON"].value_counts()

merged_data['KATEGORİ'].nunique()

merged_data.dropna(inplace=True)

merged_data.describe().T

# Scatter plot
plt.scatter(merged_data['TOPLAM SİPARİŞ ADEDİ'], merged_data['TOPLAM SİPARİŞ HACMİ'])
plt.title('TOPLAM SİPARİŞ ADEDİ ve TOPLAM SİPARİŞ HACMİ')
plt.xlabel('TOPLAM SİPARİŞ ADEDİ')
plt.ylabel('TOPLAM SİPARİŞ HACMİ')
plt.show()

# Pie chart
merged_data.groupby('KATEGORİ')['TOPLAM SİPARİŞ ADEDİ'].sum().plot(kind='pie')
plt.title('KATEGORİLERE GÖRE TOPLAM SİPARİŞ ADEDİ')
plt.ylabel('')
plt.show()

sns.scatterplot(data=merged_data, x='SON 6 AY SİPARİŞ HACMİ', y='KATEGORİ')
plt.title('SON 6 AY SİPARİŞ HACMİ ve KATEGORİ İLİŞKİSİ')
plt.xlabel('SON 6 AY SİPARİŞ HACMİ')
plt.ylabel('KATEGORİ')
plt.show()

sns.pairplot(merged_data)
plt.show()

merged_data.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(merged_data.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

plt.figure(figsize=(11,11))
sns.countplot(data=merged_data, x="KATEGORİ",palette=['#432371',"#FAAE7B"])
plt.xticks(rotation=90)
plt.show()

# Scatter Plot
sns.scatterplot(x="TOPLAM SİPARİŞ ADEDİ", y="TOPLAM SİPARİŞ HACMİ", hue="LOKASYON", style="KATEGORİ", data=merged_data)
plt.title("Toplam Sipariş Adedi ve Hacmi İlişkisi")
plt.show()

sns.countplot(x="SON 6 AY SİPARİŞ ADEDİ", hue="KATEGORİ", data=merged_data)
plt.title("Son 6 Ay Sipariş Adedi Dağılımı")
plt.show()

sns.lineplot(x="SON SİPARİŞ TARİHİ", y="SON 6 AY SİPARİŞ HACMİ", hue="LOKASYON", data=merged_data)
plt.title("Son 6 Ay Sipariş Hacmi Değişimi")
plt.show()

sns.heatmap(merged_data[["TOPLAM SİPARİŞ ADEDİ", "TOPLAM SİPARİŞ HACMİ"]].corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Isı Haritası")
plt.show()

sns.pairplot(data=merged_data, hue="LOKASYON", diag_kind="hist")
plt.suptitle("Değişkenlerin Birbirleri ile İlişkisi")
plt.show()

sns.histplot(data=merged_data, x="SON 6 AY SİPARİŞ ADEDİ", kde=True)
plt.title("Son 6 Ay Sipariş Adedi Dağılımı")
plt.show()

sns.violinplot(x="KATEGORİ", y="TOPLAM SİPARİŞ HACMİ", hue="LOKASYON", data=merged_data)
plt.title("Kategoriye Göre Toplam Sipariş Hacmi Dağılımı")
plt.show()

####################
# Machine Learning #
####################

X = merged_data.drop(["AKTİF/PASİF", "LOKASYON", "MÜŞTERI NO", "KATEGORİ", "SON SİPARİŞ TARİHİ"],axis=1)
y = merged_data[["AKTİF/PASİF"]]
X.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

import joblib
joblib.dump(encoder, "D:/avansas_case/label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_score(y_pred, y_test)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy_score(y_pred, y_test)

import joblib
joblib.dump(rf_model, "D:/avansas_case/03.randomforest_with_churn.pkl")

classifier_loaded = joblib.load("D:/avansas_case/03.randomforest_with_churn.pkl")
encoder_loaded = joblib.load("D:/avansas_case/label_encoder.pkl")

merged_data_test.dropna(inplace=True)
merged_data_test.drop(["LOKASYON", "MÜŞTERI NO", "KATEGORİ", "SON SİPARİŞ TARİHİ"],axis=1, inplace=True)

X_manual_test = merged_data_test
print("X_manual_test", X_manual_test)

prediction_raw = classifier_loaded.predict(X_manual_test)
print("prediction_raw", prediction_raw)

prediction_real = encoder_loaded.inverse_transform(classifier_loaded.predict(X_manual_test))
print("Real prediction", prediction_real)

