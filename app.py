import numpy as np
from flask import Flask,request,render_template
import pickle 
import math
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl",'rb'))
pipeline = pickle.load(open("pipeline.pkl",'rb'))
feats = ['Engine Type', 'HP or lbs thr ea engine', 'Max speed Knots',
       'Rcmnd cruise Knots', 'Stall Knots dirty', 'Fuel gal/lbs',
       'All eng rate of climb', 'Eng out rate of climb', 'Takeoff over 50ft',
       'Landing over 50ft', 'Empty weight lbs', 'Length ft/in',
       'Wing span ft/in', 'Range N.M.']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    feat = np.array([x.strip() for x in request.form.values()])
    feat[1:] = np.array(feat[1:])
    final_feat = [feat]
    feat_df = pd.DataFrame(final_feat,columns=feats)
    trans_feat = pipeline.transform(feat_df)
    pred = model.predict(trans_feat)
    output = round(pred[0],2)
    return render_template("index.html",prediction_text="Plane Price: {}".format(output))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)