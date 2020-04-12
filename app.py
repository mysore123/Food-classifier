# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:34:57 2020

@author: apmys
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, redirect,url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    else:
        return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    print("Loading model")

    global model
    model = load_model('finetune4.h5')

    #my_image = plt.imread(os.path.join('uploads', filename),format=(224,224))
    my_image=load_img(os.path.join('uploads',filename),target_size=(224,224))

    image = np.reshape(my_image,(224,224,3))
    print(image.shape)
    
    image = preprocess_input(image)
    print(image.shape)
    
   

    probabilities = model.predict(np.array( [image,] ))[0,:]
    print(probabilities)
#Step 4
    number_to_class = ['dosa', 'idli', 'mysore bonda', 'mysore pak', 'paratha', 'poha', 'rasam' ]
    index = np.argsort(probabilities)
    print(index)
    calories={"dosa":'A dosa is a cooked flat thin layered rice batter, originating from South India, made from a fermented batter. Its main ingredients are rice and black gram that are grounded together in a fine, smooth batter with a dash of salt.  Average calories:133 Carbohydrates:75 calories protein:11 calories fat:47 calories',
              "idli":'Idlis are a type of savoury rice cake originating from south India.These are made by steaming a batter consisting of fermented black lentils.  Average Calories:33  carbs:29calories  proteins:4calories fat:1calorie',
              "mysore bonda":'Bonda is a typical South Indian snack that has various sweet and spicy versions in different regions. Most common of which is Aloo Bonda and other region specific variations.  Average calories:227(1 serving) carbs:93calories protein:17calories fat:110 calories.',
              "mysore pak":'Mysore pak is an Indian sweet prepared in ghee that is popular in Southern India. It originated in the Indian state of Karnataka. It is made of generous amounts of ghee, sugar, gram flour, and often cardamom. Average Calories: 564 Carbohydrates: 68 calories, Proteins: 5 calories Fat: 491 calories',
              "paratha":'A paratha is a flatbread native to the Indian subcontinent, prevalent throughout the modern-day nations of India, Sri Lanka, Pakistan, Nepal, Bangladesh, Maldives, and Myanmar, where wheat is the traditional staple. Average Calories: 126 Carbohydrates: 63 calories, Proteins:10calories Fat: 59calories',
              "poha":'Poha (flattened rice) is an easy, delicious and healthy breakfast recipe, popular in Maharashtra. Made with onions, potatoes and seasoning like chillies, lemon and curry leaves make up a tasty and easy meal of Poha. Average Calories: 180 (1 plate), Carbohydrates: 100calories, Proteins:9calories  Fat:71 calories.',
              "rasam":'Rasam, charu pani, chaaru, saaru or kabir is a South Indian dish, traditionally prepared using kokum or tamarind juice as a base, with the addition of tomato, chili pepper, black pepper, cumin and other spices as seasonings. Average Calories:64(1 serving)',
              }
    predictions = {
        "class1":number_to_class[index[6]],
        "class2":number_to_class[index[5]],
        "class3":number_to_class[index[4]],
        "prob1":probabilities[index[6]],
        "prob2":probabilities[index[5]],
        "prob3":probabilities[index[4]],
        "cal1":calories[number_to_class[index[6]]]
      }
#Step 5

    return render_template('predict.html', predictions=predictions)