from flask import Flask, request, render_template, redirect, url_for
import hashlib
import os
import numpy as np

import csv
from scipy import signal
import sys
import math
from numpy import NaN, Inf, arange, isscalar, asarray, array
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Store users in a dictionary where the key is the username
# and the value is a tuple containing the hashed password and other information
users = {}

def hash_password(password):
    # Use a hash function to store hashed password for security
    return hashlib.sha256(password.encode()).hexdigest()

# def pass1(file_name):
#     y = []
#     with open(file_name, "r") as file:
#         y = file.read()
    

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def peakdet(v, delta, x = None):
    maxtab = []
    mintab = []
    if x is None:
        x = arange(len(v))
    v = asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab), array(mintab)



# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         username = request.form["username"]
#         password = request.form["password"]

#         # Check if the username already exists
#         if username in users:
#             return "User already exists. Please choose a different username."

#         # Validate the password
#         if len(password) < 8:
#             return "Password must be at least 8 characters long."

#         # Hash the password and store the user information
#         hashed_password = hash_password(password)
#         users[username] = (hashed_password,)

#         return redirect(url_for("login"))

#     return render_template("signup.html")





@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
             return redirect(url_for("home"))
        else:
            error = 'Invalid username or password. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    y = []
    
    if request.method == 'POST':
        file = request.files['file']
        contents = file.read().decode()
        numbers = [int(num) for num in contents.split()]
        y.extend(numbers)
        fps = 340 #Sampling rate

        filtered_sine = butter_highpass_filter(y,0.1,fps)

        filtered_sine1 = butter_lowpass_filter(filtered_sine,20,fps)

        delt=int((max(filtered_sine1)-min(filtered_sine1))*.55)



        maxtab, mintab = peakdet(filtered_sine1,delt)
       
        heart_rate1=0
        epoc_duration1=0
        ppos=0
        tpos=0
        ppos1=0
        qpos1=0
        rpos1=0
        spos1=0
        tpos1=0
        P_wave1=0
        Q_wave1=0
        qrs1=0
        R_wave1=0
        S_wave1=0
        T_wave1=0
        PR_interval1=0
        QT_interval1=0
        PR_segment1=0
        ST_segment1=0
        k=1
        for i in range(0,10):
            t=math.floor(maxtab[i][0])
            s=math.floor(maxtab[i+1][0])
            half=int((s-t)/2)
            #r wave peaks
            epoc=filtered_sine1[t+half-30:s+half+30]
            rpeak=0
            for i in range(0,len(epoc)):
                if epoc[i]>rpeak:
                    rpeak=epoc[i]
                    rpos=i

            #Q wave peaks
            i=rpos
            while(epoc[i]>epoc[i-1]):
                i=i-1
            qpeak=epoc[i]
            qpos=i#Q peak position


            #average
            avg=0
            for i in range(1,50):
                avg=epoc[i]+avg
            avg=avg/50
            
            #S wave peaks
            i=rpos
            while(epoc[i]>epoc[i+1]):
                i=i+1
            speak=epoc[i]
            spos=i
            i=qpos
            j=spos
            qr=[]
            g=0
            while(epoc[i]<epoc[i-1] and g<=9):
                i=i-1
                g=g+1
            printerval2=i
            l=0
            while(epoc[j]<epoc[j+1] and l<=9):
                j=j+1
                l=l+1
            stsegment1=j
            qrs=((j-i)/fps)*1000
            R_wave=((spos-qpos)/fps)*1000
            Q_wave=(((qpos-i)+((qpos-i-1)/2))/fps)*1000
            S_wave=(((j-spos)+((j-spos-1)/2))/fps)*1000
            while(i!=j):
                qr.append([i, epoc[i]])
                i=i+1
            
            #P wave peaks
            ppeak=0
            for i in range(50,qpos):
                if epoc[i]>ppeak:
                    ppeak=epoc[i]
                    ppos=i
            i=ppos
            j=i
            pp=[]
            if(epoc[ppos]>avg):
                while((epoc[i]>epoc[i-1] or epoc[i]>epoc[j+8] )and i>(ppos-25)):
                    pp.append([i,epoc[i]])
                    i=i-1
                while(((epoc[j]>epoc[j+1] or epoc[j]>epoc[j+8])and i<(ppos+24)) and j<(printerval2-2)):
                    pp.append([j,epoc[j]])
                    j=j+1
            else:
                i=0
                j=0
                pp.append([0,0])
            printerval1=i
            prsegment=j
            P_wave=((j-i)/fps)*1000

            #T wave peaks
            tpeak=0
            for i in range(spos,len(epoc)):
                if epoc[i]>tpeak:
                    tpeak=epoc[i]
                    tpos=i
            i=tpos
            j=i
            tt=[]
            if(epoc[tpos]>avg):
                while(((epoc[i]>epoc[i-1] or epoc[i]>epoc[j+9] )and i>(tpos-55)) and i>(stsegment1-2) ):
                    tt.append([i,epoc[i]])
                    i=i-1
                while((epoc[j]>epoc[j+1] or epoc[j]>epoc[j+7])and i<(tpos+28)):
                    tt.append([j,epoc[j]])
                    j=j+1
            else:
                i=0
                j=0
                tt.append([0,0])
            
            qtinterval=j
            stsegment2=i
            T_wave=((j-i)/fps)*1000
            v=(s-t)/fps
            PR_interval=((printerval2-printerval1)/fps)*1000
            QT_interval=((qtinterval-printerval2)/fps)*1000
            PR_segment=((printerval2-prsegment)/fps)*1000
            ST_segment=((stsegment2-stsegment1)/fps)*1000
            heart_rate=int(60/v)
            epoc_duration=int(v*1000)
            

            example_ECG1 = [[heart_rate, epoc_duration,ppos,qpos,rpos,spos,tpos, P_wave, Q_wave, qrs, R_wave,S_wave, T_wave, PR_interval, QT_interval, PR_segment, ST_segment]]
            
            heart_rate1=(heart_rate1*(k-1)+heart_rate)/k
            epoc_duration1=(epoc_duration1*(k-1)+epoc_duration)/k
            ppos1=(ppos1*(k-1)+ppos)/k
            qpos1=(qpos1*(k-1)+qpos)/k
            rpos1=(rpos1*(k-1)+rpos)/k
            spos1=(spos1*(k-1)+spos)/k
            tpos1=(tpos1*(k-1)+tpos)/k
            P_wave1=(P_wave1*(k-1)+P_wave)/k
            Q_wave1=(Q_wave1*(k-1)+Q_wave)/k
            qrs1=(qrs1*(k-1)+qrs)/k
            R_wave1=(R_wave1*(k-1)+R_wave)/k
            S_wave1=(S_wave1*(k-1)+S_wave)/k
            T_wave1=(T_wave1*(k-1)+T_wave)/k
            PR_interval1=(PR_interval1*(k-1)+PR_interval)/k
            QT_interval1=(QT_interval1*(k-1)+QT_interval)/k
            PR_segment1=(PR_segment1*(k-1)+PR_segment)/k
            ST_segment1=(ST_segment1*(k-1)+ST_segment)/k
            print(S_wave1)
            heart_data = pd.read_csv('final.csv')
           



            # checking for missing values
            heart_data.isnull().sum() 
            X = heart_data.drop(columns='class', axis=1)
            Y = heart_data['class']



            X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=42)



            model = LogisticRegression()
            model.fit(X_train, Y_train)


            input_data = (P_wave, qrs, T_wave, heart_rate, QT_interval, PR_interval)

            # change the input data to a numpy array
            input_data_as_numpy_array= np.asarray(input_data)

            # reshape the numpy array as we are predicting for only on instance
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

            prediction = model.predict(input_data_reshaped)
            if (prediction[0]== 0):
                cond="Normal ECG"
            else:
                cond="Abnormal ECG"
                

        return render_template("index.html", heart_rate1= heart_rate1,S_wave1=S_wave1,P_wave1=P_wave1, Q_wave1= Q_wave1,qrs1=qrs1,R_wave1=R_wave1,T_wave1=T_wave1,PR_interval1=PR_interval1,QT_interval1=QT_interval1,cond=cond)
    return render_template("index.html")




if __name__=="__main__":
    app.run(debug=True, port=7000)