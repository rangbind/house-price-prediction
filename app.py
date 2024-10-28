from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load the saved model
model = joblib.load('house_price_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize the scaler and encoder
# scaler = StandardScaler()
# encoder = OneHotEncoder(drop='first')

# Route 1: Home Page
@app.route('/')
def home():
    return render_template('home.html')  # Shows "Welcome to home page"

# Route 2: Get Prediction - shows form
@app.route('/get_prediction')
def get_prediction():
    return render_template('get_prediction.html')  # Shows form for input

# Route 3: Post Prediction - receives form data and predicts price
@app.route('/post_prediction', methods=['POST'])
def post_prediction():
    # Get form data
    sqft = request.form.get('SqFt')
    bedrooms = request.form.get('Bedrooms')
    bathrooms = request.form.get('Bathrooms')
    offers = request.form.get('Offers')
    brick = request.form.get('Brick')
    neighborhood = request.form.get('Neighborhood')
    
    # Create DataFrame for model prediction
    new_data = pd.DataFrame({
        'SqFt': [float(sqft)],
        'Bedrooms': [int(bedrooms)],
        'Bathrooms': [int(bathrooms)],
        'Offers': [int(offers)],
        'Brick': [brick],
        'Neighborhood': [neighborhood]
    })

    categorical_feature_2 = encoder.transform(new_data[['Brick', 'Neighborhood']]).toarray()
    numerical_feature_new = new_data[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers']]
    numerical_feature_2 = scaler.transform(numerical_feature_new)
    X_scalled_new = np.hstack((numerical_feature_2, categorical_feature_2))
    predicted_price = model.predict(X_scalled_new)
    predicted_price = int(predicted_price[0])
    # Render result template with prediction
    return render_template('prediction_result.html', price=predicted_price)  # Display first prediction result
    # return render_template('prediction_result.html', price=100)  # Display first prediction result

if __name__ == '__main__':
    app.run(debug=False)
