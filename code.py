import pandas as pd
import numpy as np
jiance1 = pd.read_excel('监测1.xls')
jiance2 = pd.read_excel('监测2.xls')
jiance3 = pd.read_excel('监测3.xls')
jiance4 = pd.read_excel('监测4.xls')
qixiang1 = pd.read_excel('气象1.xls',index_col=None)
qixiang2 = pd.read_excel('气象2.xls',index_col=None)
qixiang3 = pd.read_excel('气象3.xls',index_col=None)
qixiang4 = pd.read_excel('气象4.xls',index_col=None)  # 源数据读取

jiance_data = jiance1.append(jiance2)
jiance_data = jiance_data.append(jiance3)
jiance_data = jiance_data.append(jiance4)  # 数据归并

qixiang_data = qixiang1.append(qixiang2)
qixiang_data = qixiang_data.append(qixiang3)
qixiang_data = qixiang_data.append(qixiang4)  # 数据归并

qixiang_data.drop_duplicates(['站点名称','时间'],inplace=True)
jiance_data.drop_duplicates(['站点名称','时间'],inplace=True)  # 数据去重

zdmc = list(set(list(jiance_data['站点名称']))) # 站点名称

final_data = pd.merge(qixiang_data, jiance_data, how='inner') #拼接气象数据以及监测数据
final_data.drop(['时间'],axis=1,inplace=True) # 删除“时间”列（对后续数据计算没有什么影响）

each_data = final_data[final_data['站点名称']=='浑南东路'] # 选出一个站点的数据进行试验
each_data.fillna(method='pad',inplace=True)  # 填补空值
y_data = each_data['PM2.5']
x_data = each_data.drop(['站点名称','PM2.5'],axis=1)

# 数据预处理
x_data = x_data.fillna(method='bfill') 
x_data_gy = (x_data - x_data.min())/(x_data.max()-x_data.min())
x_data = x_data_gy.values
y_data_2 = (y_data-y_data.min())/(y_data.max()-y_data.min())
y_data.reset_index(drop=True,inplace=True)
# x_data2 = np.ones((7265,36))
x_data2 = np.zeros((7303, 3, 13))
for i in range(7303):
    for j in range(3):
        x_data2[i][j] = np.hstack((x_data[i+j],y_data[i+j]))
        # x_data2[i] = np.append(x_data[i],y_data[i:i+21], axis=0)
y_data_3 = y_data_2[10:]   


# emd分解
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
decomposer = EMD(y_data_3)
imfs = decomposer.decompose()
# y_data_3 = y_data[31:]   

# 多层小波分解
import pywt
coeffs = pywt.wavedec(y_data_3, 'db1' ,level=3)
# coeffs[0] = np.zeros_like(coeffs[0])
coeffs[1] = np.zeros_like(coeffs[1])
coeffs[2] = np.zeros_like(coeffs[2])
coeffs[3] = np.zeros_like(coeffs[3])
data_cD_0 = pywt.waverec(coeffs, 'db1')

#arima
from statsmodels.tsa.arima_model import ARIMA 
y_data = list(y_data)
pre_ar = []
test_ar = []
import random
for i in range(1000, 3000):
    # j = random.randint(2000,7296)
    # test_ar.append(y_data[j+6])
    model_ar = ARIMA(y_data[i-1000:i],(1,1,1)).fit()
    pre_ar.append(model_ar.forecast(7)[0][6])
  
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split # 划分训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(x_data2,y_data_3,test_size = 0.3,random_state = 0)

# 深度学习模型构建
from keras import metrics
def model():
    model = Sequential()
    model.add(LSTM(units=128, activation='tanh', return_sequences=True, batch_input_shape=(None,3,13))) #  return_sequences=True,
    model.add(LSTM(units=64, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='sum_mean_squared_error', optimizer='adam', metrics=['smse'])
    return model
model = model()

model.fit(train_x, train_y, epochs=500, batch_size=64, verbose=1)
pre_y = model.predict(test_x)

pre_y2 = []
for i in range(2180):
    pre_y2.append(sum(pre_y[i]) * (y_data.max()-y_data.min()))
    
test_y = list(test_y)
test_y2 = []
for i in range(2180):
    test_y2.append(sum(test_y[i]) * (y_data.max()-y_data.min()))
    
import copy
pre_wd = copy.deepcopy(pre_y2) # wd-lstm预测值
pre_emd = copy.deepcopy(pre_y2) # emd-lstm预测值 
pre_lstm = copy.deepcopy(pre_y2) # lstm预测值
test = copy.deepcopy(test_y2) # 真实值
pre_xgb = copy.deepcopy(pre_y2) # xgboost模型预测值
pre_arima = copy.deepcopy(pre_ar) # arima模型预测值

from matplotlib.pylab import style
style.use('seaborn-ticks')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False 

# 画图
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
# plt.subplot(2,1,1)
# ax = plt.subplot(2,1,1)
plt.plot(range(10), test[250:260], label='原PM2.5序列',c='red')
plt.plot(range(10), pre_emd[250:260], label='MRMR-EMD-LSTM',c='blue', marker='o',ms=2,markersize=7)
plt.plot(range(10), pre_wd[250:260], label='WD-LSTM',c='black', marker='+',ms=2,markersize=7)
plt.plot(range(10), pre_lstm[250:260], label='LSTM',c='black', marker='*',ms=2,markersize=7)
plt.plot(range(10), pre_xgb[250:260], label='Xgboost',c='black', marker='x',ms=2,markersize=7)
plt.legend(fontsize='xx-large')

# 评价指标
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
rmse = np.sqrt(mean_squared_error(pre_y2, test_y2))
mae = mean_absolute_error(pre_y2, test_y2)
r2 = r2_score(pre_y2, test_y2)
plt.title("预测结果拟合",fontsize='xx-large')
