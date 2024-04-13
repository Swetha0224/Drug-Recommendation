import joblib
from flask import Flask, render_template, request, redirect, session,request,jsonify,url_for
import os
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from pymongo import MongoClient
from flask_pymongo import PyMongo

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/drugs'

mongo = PyMongo(app)


app.secret_key=os.urandom(24)
# Model saved with Keras model.save()
MODEL_PATH = 'passmodel.pkl'

TOKENIZER_PATH ='tfidfvectorizer.pkl'

DATA_PATH ='data/drugsComTrain.csv'

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
model = joblib.load(MODEL_PATH)

#getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def is_logged_in():
    return 'user_id' in session

@app.route('/')
def login():
	return render_template('main.html')
@app.route('/about')
def about():
    return render_template('about.html')
  

@app.route("/logout")
def logout():
	session.clear()
	return redirect('/')

@app.route('/index1')
def index1():
    if 'user_id' in session:
        return render_template('main.html')
    else:
        return redirect('/')

@app.route('/index')
def index():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')
     
        
	

@app.route('/register',methods=['POST','GET'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username already exists in the database
        existing_user = mongo.db.users.find_one({'username': username})
        if existing_user:
            return "Username already exists! Please choose another one."
        
        # Insert the new user into the database
        mongo.db.users.insert_one({'username': username, 'password': password})
        
        # Redirect to login page after successful registration
        return render_template('login.html')
    return render_template('login.html')

@app.route('/login_validation', methods=['GET','POST'])
def login_validation():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if the username and password match a user in the database
        user = mongo.db.users.find_one({'username': username, 'password': password})
        if user:
            # Store user information in the session
            session['user_id'] = str(user['_id'])
            session['username'] = username
            return redirect('/index1')  # Redirect to the home page or wherever you want
                
        else:
            err = "Invalid username or password"
            return render_template('login.html', lbl=err)
        
    return render_template('register.html')


@app.route("/book")
def book():
    return redirect("https://www.apollo247.com/specialties")

@app.route('/predict1')
def predict1():
    if 'user_id' in session:
        return redirect('/index')
    else:
        return redirect('/login_validation')
        
		
		
# 		return redirect('/index')
# 	else:
# 		return redirect('/')   
    
# @app.route('/predict')
# def predict():
# 	if 'user_id' in session:
		
		
# 		return redirect('/index')
# 	else:
# 		return redirect('/')
    

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        if 'user_id' in session:
            raw_text = request.form.get('rawtext', '')  # Get 'rawtext' from form data or assign empty string if not provided
        
        # Check if raw_text is empty or consists only of whitespace
            if raw_text.strip() == '':
                raw_text = "There is no text to select"
                return jsonify({'error': 'Empty raw text provided'})
        
        # If raw_text is not empty, proceed with prediction
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]
	
            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)
            if 'user_id' in session:
            
                data = {
                    'rawtext': raw_text,
                    'rawtext1': clean_text,
                    'result': predicted_cond,
                    'top_drugs': top_drugs,
                   
                }
            # Insert data into MongoDB collection
            # mongo.db.users.insert_one(data)
            user_data = mongo.db.users.find_one({'username': session['username']})
            if user_data and 'data_items' in user_data and isinstance(user_data['data_items'], list):
       
                result = mongo.db.users.update_one(
                {'username': session['username']},
                {'$addToSet': {'data_items': data}}
                )
            else:
        
                result = mongo.db.users.update_one(
                {'username': session['username']},
                {'$set': {'data_items': [data]}}
                )
            
            # mongo.db.users.update_one(
            #     {'username': session['username']},  # Filter by user_id
            #     {'$set': data},  # Update fields with new data
            #     upsert=True  # Insert document if it doesn't exist
            # )
            
        # print(tfidf_vect)
        return render_template('predict.html', rawtext1=clean_text,rawtext=raw_text, result=predicted_cond, top_drugs=top_drugs)
        
    return render_template('login.html')



@app.route('/history')
def history():
    if 'username' in session:
        # Retrieve user information and data items associated with the current user from MongoDB
        user_data = mongo.db.users.find_one({'username': session['username']})
        username = session['username']
        data_items = user_data.get('data_items', []) if user_data else []

        # Pass data items and username to the template for rendering
        return render_template('history.html', data_items=data_items, username=username)
    else:
        return redirect('/login')  # Redirect to the login page if user is not logged in


def cleanText(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst




# print(users)


if __name__ == "__main__":
	
	app.run(debug=True, host="localhost", port=8080)