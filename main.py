print("===== STUDENT PERFORMANCE PREDICTOR =====")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



data =pd.read_csv("student_data.csv")
print(data.columns)
print(data.head())

X=data[['studyhours','attendance','assignments']]
y=data['marks']

X_train,X_test,y_train,y_test =train_test_split(
    X,y,test_size=0.2,random_state=42
)

model= LinearRegression()
model.fit(X_train,y_train)

study_hours =float(input("enter your study hours:"))
attendance=float(input("enter attendance(%): "))
assignments =int(input("enter assignments compeleted: "))


prediction=model.predict([[study_hours,attendance,assignments]])

print("\npredicted marks :",round(prediction[0],2))

if prediction[0] >=90:
    print ("grade:A+=")
elif prediction[0]>=75: 
    print("grade: A")
elif prediction[0]>=60:
    print("grade:B")
else:print("grade:C")

accuracy = model.score(X_test,y_test)
print("model accuracy:",round(accuracy *100, 2),"%")

with open("results.txt","a")as f:
    f.write(
        f"study hours={study_hours}, "
        f"attendamnce={attendance},  "
        f"assignment={assignments}, "
        f"predicted marks={round(prediction[0],2)}\n"

    )

print("results saved to results.txt")

plt.scatter(data['studyhours'],data['marks'])
plt.xlabel("study hours")
plt.ylabel("marks")
plt.title("student performance predictor")
plt.savefig("graph.png")
print("graph saved as graph.png")