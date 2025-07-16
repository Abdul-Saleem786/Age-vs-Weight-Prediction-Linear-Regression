import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Example data:Months vs sales
age=np.array([5,10,15,20]).reshape(-1, 1)
weight=np.array([10,20,30,40]).reshape(-1, 1)

#Create and train Model
model=LinearRegression()
model.fit(age,weight)

#Predict Future Months
future_age=np.array([25,30]).reshape(-1, 1)
predicted_weight=model.predict(future_age)

#Outpu Predict
for i, age_num in enumerate(future_age.flatten(),start=1):
    print(f"Predicted weight for age {age_num}:{predicted_weight[i-1]}")

plt.scatter(age.flatten(),weight.flatten(), color="blue")
plt.plot(age.flatten(),model.predict(age).flatten(),color='green')
plt.plot(future_age.flatten(),predicted_weight.flatten(), color='red')
plt.xlabel("Age")
plt.ylabel("weight")
plt.title("weight Prediction")
plt.show()