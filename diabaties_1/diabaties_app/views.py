from django.shortcuts import render,redirect

import numpy as np
import joblib





def index(request):
    return render(request,'index.html')


def predict(request):
    if request.method == "POST":
        classifier=joblib.load('finalized_model.sav')

        preg = int(request.POST.get('pregnancies'))
        glucose = int(request.POST.get('glucose'))
        bp = int(request.POST.get('bloodpressure'))
        st = int(request.POST.get('skinthickness'))
        insulin = int(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        dpf = float(request.POST.get('dpf'))
        age = int(request.POST.get('age'))

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render(request,'result.html', {'prediction':my_prediction})


