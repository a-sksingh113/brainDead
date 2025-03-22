from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

model_path = 'ipl_model.pkl'  
with open(model_path, 'rb') as file:
    model = pickle.load(file)
app = Flask(__name__)
team_mapping = {
    'Chennai Super Kings': 0, 'Delhi Capitals': 1, 'Gujarat Titans': 2, 'Kolkata Knight Riders': 3,
    'Lucknow Super Giants': 4, 'Mumbai Indians': 5, 'Punjab Kings': 6, 'Rajasthan Royals': 7,
    'Royal Challengers Bangalore': 8, 'Sunrisers Hyderabad': 9
}
venue_mapping = {
    'Wankhede Stadium': 0, 'Eden Gardens': 1, 'M. A. Chidambaram Stadium': 2, 'Narendra Modi Stadium': 3
}
city_mapping = {
    'Mumbai': 0, 'Kolkata': 1, 'Chennai': 2, 'Ahmedabad': 3
}
toss_decision_mapping = {'bat': 0, 'field': 1}
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/ipl')
def ipl():
    return render_template('ipl.html')
@app.route('/ipl/predict', methods=['POST'])
def predict_ipl():
    try:
        # Extract data from form
        data = request.form.to_dict()

        # Convert categorical values to numerical
        features = [
            team_mapping.get(data['team1'], -1),
            team_mapping.get(data['team2'], -1),
            venue_mapping.get(data['venue'], -1),
            city_mapping.get(data['city'], -1),
            team_mapping.get(data['toss_winner'], -1),
            toss_decision_mapping.get(data['toss_decision'], -1)
        ]
        if -1 in features:
            return jsonify({'error': 'Invalid input values'})
        final_features = np.array([features]).astype(float)

        prediction = model.predict(final_features)[0]
        predicted_winner = [team for team, num in team_mapping.items() if num == prediction][0]
        return render_template('ipl.html', prediction_text=f'Predicted Winner: {predicted_winner}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
