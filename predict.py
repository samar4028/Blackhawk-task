import os
import pandas as pd

import joblib
import config
import json
from flask import Flask, request, jsonify
app = Flask(__name__)



@app.route("/predict",  methods = ['POST'])
def my_prediction():
    content = request.data
    #print(content)

    json_dict = json.loads(content)
    #print(json_dict)

    df = pd.DataFrame.from_dict(json_dict)
    #print(df)
    results_ = joblib.load(filename=config.PIPELINE_NAME)

    results = results_.predict(df['input'])
    #print(results)
    return json.dumps(results.tolist())
   # df['res']='spam'
   # print(df)

   # return df.to_json(orient ='records')


# def make_prediction(input_data):
    
#     _text_clf_lsvc = joblib.load(filename=config.PIPELINE_NAME)
    
#     results = _text_clf_lsvc.predict(input_data)

#     return results



if __name__ == '__main__':
    
    # test pipeline
    import numpy as np

    #input_data1 = input()
    #input_data2 = pd.DataFrame([input_data1])
    
    
    #pred = make_prediction(input_data2[0])
    #print(pred[0])

    app.run(host='0.0.0.0', port=5000)


