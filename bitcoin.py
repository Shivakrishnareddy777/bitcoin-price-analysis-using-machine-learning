
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


training_set=pd.read_csv('BTCtrain.csv')      
training_set.head()                           
training_set1=training_set.iloc[:,1:2]        
training_set1.head()                         
training_set1=training_set1.values            
training_set1                                 


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()                          
training_set1 = sc.fit_transform(training_set1)
xtrain=training_set1[0:2694]                 		   
ytrain=training_set1[1:2695]                 

today=pd.DataFrame(xtrain)              
tomorrow=pd.DataFrame(ytrain)           
ex= pd.concat([today,tomorrow],axis=1)       
ex.columns=(['today','tomorrow'])
xtrain = np.reshape(xtrain, (2694, 1, 1))     

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


regressor=Sequential()                                                     

regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))      
regressor.add(Dense(units=1))                                               

regressor.compile(optimizer='adam',loss='mean_squared_error')               

regressor.fit(xtrain,ytrain,batch_size=32,epochs=2000)                      

test_set = pd.read_csv('BTCtest.csv')
test_set.head()


real_stock_price = test_set.iloc[:,1:2]         

real_stock_price = real_stock_price.values      

inputs = real_stock_price			
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (8, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

plt.plot(real_stock_price, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.legend()
plt.show()



regressor.fit(xtrain,ytrain,batch_size=32,epochs=2000)                     