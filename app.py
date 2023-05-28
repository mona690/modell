# import warnings

# import keras

# warnings.filterwarnings('ignore')
# # Data processing
# import pandas as pd
# import math
# import numpy as np
# import librosa
# import os
# from collections import Counter
# # Visualization
# import matplotlib.pyplot as plt
# import seaborn as sns
# import librosa.display
# # Model and performance
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# #from tensorflow.keras import keras
# from sklearn.metrics import classification_report,mean_absolute_error
# #from tensorflow.keras.applications.mobilenet import preprocess_input
# from werkzeug.utils import secure_filename
# flask
from flask import Flask, request, send_file, jsonify

filepath = None

# In[ ]:


app = Flask(__name__)


# # In[ ]:


# import pickle

# # Load the first model from the .pkl file
# model_1 = tf.keras.models.load_model('LSTM.h5')
# # Load the second model from the .pkl file
# model_2 = tf.keras.models.load_model('AutoEncoder.h5')


# # In[ ]:
# def auto_feature_extraction(file_path, num_mfcc=40, n_fft=2048, hop_length=2048, num_segment=1):
#     mfccs = []
#     sample_rate = 44100
#     samples_per_segment = int(sample_rate * 1 / num_segment)
#     try:
#         y, sr = librosa.load(file_path, sr=sample_rate, res_type='kaiser_fast')
#         for n in range(num_segment):
#             mfcc = librosa.feature.mfcc(y=y[samples_per_segment * n: samples_per_segment * (n + 1)],
#                                         sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
#                                         hop_length=hop_length)
#             mfcc = np.mean(mfcc.T, axis=0)
#             mfccs.append(mfcc.tolist())
#     except:
#         pass
#     return np.array(mfccs).reshape(1, -1)


# # In[ ]:
# def lstm_preprocess(file_path, num_mfcc=40, n_fft=2048, hop_length=1024, num_segment=1):
#     mfccs = []
#     sample_rate = 44100
#     samples_per_segment = int(sample_rate / num_segment)
#     if file_path.endswith('.wav'):
#         try:
#             y, sr = librosa.load(file_path, sr=sample_rate, res_type='kaiser_fast')
#         except:
#             return None
#         for n in range(num_segment):
#             mfcc = librosa.feature.mfcc(y=y[samples_per_segment * n: samples_per_segment * (n + 1)], sr=sr,
#                                          n_mfcc=num_mfcc, n_fft=n_fft,
#                                          hop_length=hop_length)
#             mfcc = mfcc.T
#             max_length = 44
#             print(len(mfcc))
#             if (len(mfcc) > max_length):
#                 mfcc = mfcc[:, :max_length]
#             elif (len(mfcc) < max_length):
#                 pad_width = max_length - len(mfcc)
#                 mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='edge')
#             mfccs.append(mfcc.tolist())
#     return mfccs


# In[ ]:
@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    # Get file path from request body
    data = request.get_json()
    filepath = data.get('filepath')
    if not filepath:
        return 'No file path provided'
    else:
        return 'path received'

    # # Handle prediction
    # data_lstm = lstm_preprocess(filepath)
    # data_auto = auto_feature_extraction(filepath)
    # x_lstm = np.asarray(data_lstm[0]).astype(np.int64)
    # x = x_lstm.reshape(1, 44, 40)
    # y_auto = model_2.predict(data_auto)
    # mae = mean_absolute_error(data_auto, y_auto)
    # result = {'loss': mae}
    # if mae <= 7.3:
    #     predict_y = model_1.predict(x)
    #     predict_y = np.argmax(predict_y, axis=1)
    #     result['status'] = 'Correct Record'
    #     result['prediction'] = predict_y.tolist()
    # else:
    #     result['status'] = 'InCorrect Record'

    # return jsonify(result)


# In[ ]:

#app.run()

# %%