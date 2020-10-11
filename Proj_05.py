import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from pylab import rcParams

data = pd.read_csv('Data/Historical Product Demand.csv', parse_dates = ['Date'], index_col = ['Date'])
data.sort_values(by=['Date'], inplace=True)
data2 = data[data.index.notnull()]
data2['Order_Demand'] = data2['Order_Demand'].replace('[(]', '-', regex=True).astype(str)
data2['Order_Demand'] = data2['Order_Demand'].replace('[)]', '', regex=True).astype(int)

# plot order demand for all products
plt.figure(figsize=(20,20))
plt.xlabel('Date')
plt.ylabel('Order Demand')
plt.plot(data2['Order_Demand'])
plt.show()

sns.countplot(y='Product_Code',data=data2)
sns.countplot(y='Warehouse',data=data2)
sns.countplot(y='Product_Category',data=data2)

code_most = data2.loc[data2['Product_Code'] == 'Product_1359']['Order_Demand']
warehouse_most = data2.loc[data2['Warehouse'] == 'Whse_J']['Order_Demand']
category_most = data2.loc[data2['Product_Category'] == 'Category_019']['Order_Demand']

rcParams['figure.figsize'] = 20,10
code_model = ARIMA(code_most, order=(2,1,0))
code_result = code_model.fit()
print(code_result.summary())

code_result.plot_predict(dynamic = False)
plt.show()

rcParams['figure.figsize'] = 20,10
code_residuals = pd.DataFrame(code_result.resid)
code_residuals.plot(title="Residuals")
plt.show()

warehouse_model = ARIMA(warehouse_most, order=(2,1,0))
warehouse_result = warehouse_model.fit()
print(warehouse_result.summary())

rcParams['figure.figsize'] = 20,10
warehouse_result.plot_predict(dynamic = False)
plt.show()

rcParams['figure.figsize'] = 20,10
warehouse_residuals = pd.DataFrame(warehouse_result.resid)
warehouse_residuals.plot(title="Residuals")
plt.show()

category_model = ARIMA(category_most, order=(2,1,0))
category_result = category_model.fit()
print(category_result.summary())

rcParams['figure.figsize'] = 20,10
category_result.plot_predict(dynamic = False)
plt.show()

rcParams['figure.figsize'] = 20,10
category_residuals = pd.DataFrame(category_result.resid)
category_residuals.plot(title="Category Residuals")
plt.show()