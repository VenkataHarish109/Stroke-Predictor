import os 
from flask import Flask,render_template,request,redirect ,url_for
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,classification_report
import pandas as pd 
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
          return render_template('home.html')
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        user_name=request.form['user_name']
        password=request.form['password']
        if user_name=='harish' and password=='harish':
            return redirect(url_for('prediction'))
        else:
            return f'invalid data user_name:{user_name} OR password:{password}'
    return render_template('login.html')
@app.route('/sinin',methods=["POST",'GET'])
def sinin():
    return 'hghkghgjghkgjhg'
@app.route('/prediction',methods=['POST','GET'])
def prediction():
     if request.method =='POST':
        Gender=request.form['gender']
        Age=request.form['age']
        Hypertension=request.form['hypertension']
        Heart_Disease=request.form['heart_disease']
        Ever_Married=request.form['ever_married']
        Work_Type=request.form['work_type']
        Residence_Type=request.form['residence_type']
        Average_Glucose_Level=request.form['avg_glucose_level']
        Bmi =request.form['bmi']
        Smoking_Status=request.form['smoking_status']
        user_input={ 
        'gender': Gender ,
        'age': Age,
        'hypertension': Hypertension,
        'heart_disease': Heart_Disease,
        'ever_married': Ever_Married,
        'work_type': Work_Type,
        'Residence_type': Residence_Type,
        'avg_glucose_level':Average_Glucose_Level,
        'bmi': Bmi,
        'smoking_status': Smoking_Status}
                                                                           
        csv_file_path=os.path.join(app.root_path,'data','brain.csv')
        data = pd.read_csv(csv_file_path)
        data=data.drop('id',axis=1)
        data['bmi'].fillna(data['bmi'].mean(),inplace=True)
        data = pd.get_dummies(data) #drop_first=True
        #saparate feactures and targer 
        x=data.drop('stroke',axis=1)
        y=data['stroke']
        #split data into train and test data sets
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        model=LogisticRegression()
        #train the model 
        model.fit(x_train,y_train) 
        #prediction 
        y_pred=model.predict(x_test) 
        # Evaluation
        cm=confusion_matrix(y_test,y_pred)
        accuracy = accuracy_score(y_test, y_pred)
       #now user data predciction
        user_df = pd.DataFrame([user_input])
        user_df = pd.get_dummies(user_df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
        user_df = user_df.reindex(columns=x.columns, fill_value=0)
        prediction = model.predict(user_df)
        if prediction== 0 :
            ans='your safe'
        else:
            ans='stroke'
        return render_template('prediction.html',cm=cm,prediction=ans ,user=user_input)
     return render_template('form.html')
    
if __name__ =='__main__':
   app.run(debug=True) 