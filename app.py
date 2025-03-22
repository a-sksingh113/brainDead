from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load Model with Error Handling
model_path = 'ipl_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if model loading fails

app = Flask(__name__)

# Define Mappings
team_mapping = {
    'Chennai Super Kings': 0, 'Delhi Capitals': 1, 'Gujarat Lions': 2, 'Gujarat Titans': 3,
    'Kochi Tuskers Kerala': 4, 'Kolkata Knight Riders': 5, 'Lucknow Super Giants': 6,
    'Mumbai Indians': 7, 'Pune Warriors': 8, 'Punjab Kings': 9, 'Rajasthan Royals': 10,
    'Rising Pune Supergiant': 11, 'Royal Challengers Bengaluru': 12, 'Sunrisers Hyderabad': 13
}
city_mapping = {  # Shortened for readability
   'Abu Dhabi': 0, 'Ahmedabad': 1, 'Bangalore': 2, 'Bengaluru': 3, 'Bloemfontein': 4,
    'Cape Town': 5, 'Centurion': 6, 'Chandigarh': 7, 'Chennai': 8, 'Cuttack': 9,
    'Delhi': 10, 'Dharamsala': 11, 'Dubai': 12, 'Durban': 13, 'East London': 14,
    'Guwahati': 15, 'Hyderabad': 16, 'Indore': 17, 'Jaipur': 18, 'Johannesburg': 19,
    'Kanpur': 20, 'Kimberley': 21, 'Kochi': 22, 'Kolkata': 23, 'Lucknow': 24,
    'Mohali': 25, 'Mumbai': 26, 'Nagpur': 27, 'Navi Mumbai': 28, 'Port Elizabeth': 29,
    'Pune': 30, 'Raipur': 31, 'Rajkot': 32, 'Ranchi': 33, 'Sharjah': 34, 'Visakhapatnam': 35
}
venue_mapping = {  # Shortened for readability
     'Arun Jaitley Stadium': 0, 'Arun Jaitley Stadium, Delhi': 1, 'Barabati Stadium': 2,
    'Barsapara Cricket Stadium, Guwahati': 3, 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 4,
    'Brabourne Stadium': 5, 'Brabourne Stadium, Mumbai': 6, 'Buffalo Park': 7, 'De Beers Diamond Oval': 8,
    'Dr DY Patil Sports Academy': 9, 'Dr DY Patil Sports Academy, Mumbai': 10,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 11,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 12,
    'Dubai International Cricket Stadium': 13, 'Eden Gardens': 14, 'Eden Gardens, Kolkata': 15,
    'Feroz Shah Kotla': 16, 'Green Park': 17, 'Himachal Pradesh Cricket Association Stadium': 18,
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 19, 'Holkar Cricket Stadium': 20,
    'JSCA International Stadium Complex': 21, 'Kingsmead': 22, 'M Chinnaswamy Stadium': 23,
    'M Chinnaswamy Stadium, Bengaluru': 24, 'M.Chinnaswamy Stadium': 25, 'MA Chidambaram Stadium': 26,
    'MA Chidambaram Stadium, Chepauk': 27, 'MA Chidambaram Stadium, Chepauk, Chennai': 28,
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 29,
    'Maharashtra Cricket Association Stadium': 30, 'Maharashtra Cricket Association Stadium, Pune': 31,
    'Narendra Modi Stadium, Ahmedabad': 32, 'Nehru Stadium': 33, 'New Wanderers Stadium': 34,
    'Newlands': 35, 'OUTsurance Oval': 36, 'Punjab Cricket Association IS Bindra Stadium': 37,
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 38,
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 39,
    'Punjab Cricket Association Stadium, Mohali': 40, 'Rajiv Gandhi International Stadium': 41,
    'Rajiv Gandhi International Stadium, Uppal': 42,
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 43, 'Sardar Patel Stadium, Motera': 44,
    'Saurashtra Cricket Association Stadium': 45, 'Sawai Mansingh Stadium': 46,
    'Sawai Mansingh Stadium, Jaipur': 47, 'Shaheed Veer Narayan Singh International Stadium': 48,
    'Sharjah Cricket Stadium': 49, 'Sheikh Zayed Stadium': 50, "St George's Park": 51,
    'Subrata Roy Sahara Stadium': 52, 'SuperSport Park': 53,
    'Vidarbha Cricket Association Stadium, Jamtha': 54, 'Wankhede Stadium': 55,
    'Wankhede Stadium, Mumbai': 56, 'Zayed Cricket Stadium, Abu Dhabi': 57
}
toss_decision_mapping = {'bat': 0, 'field': 1}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ipl')
def ipl():
    return render_template('ipl.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/ipl/predict', methods=['POST'])
def predict_ipl():
    try:
        if not model:
            return jsonify({'error': 'Model not loaded. Check logs.'})

        data = request.form.to_dict()

        # Extract Features Safely
        features = [
            team_mapping.get(data.get('team1'), -1),
            team_mapping.get(data.get('team2'), -1),
            venue_mapping.get(data.get('venue'), -1),
            city_mapping.get(data.get('city'), -1),
            team_mapping.get(data.get('toss_winner'), -1),
           toss_decision_mapping.get(data.get('toss_decision', '').strip().lower(), -1)

        ]
        print("Mapped Features:", features)  
        # Validate Input
        if -1 in features:
            return jsonify({'error': 'Invalid input values. Check team, venue, city, or toss decision.'})

        # Predict Winner
        final_features = np.array([features]).astype(float)
        prediction = model.predict(final_features)[0]

        # Validate Prediction
        predicted_winner = next((team for team, num in team_mapping.items() if num == prediction), None)
        if not predicted_winner:
            return jsonify({'error': 'Unexpected prediction output.'})

        return render_template('ipl.html', prediction_text=f'Predicted Winner: {predicted_winner}')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
