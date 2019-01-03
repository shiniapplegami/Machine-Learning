from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
print(y_pred)

#Make sure the X and y are 2-D arrays. 
#For csv files with only two columns, X and y are 1D arrays and have to be converted into 2D arrays before execution

import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("X vs Y")
plt.xlabel("XLabel")
plt.ylabel("YLabel")
plt.show()
