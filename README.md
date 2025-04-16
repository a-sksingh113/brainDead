A Complete EDA of IPL History and IPL 2025 Winner Prediction [ live https://dataavengers.onrender.com ]

Problem Statement
The goal of this project is to Perform an end to end EDA and predict the winner of the 2025 Indian Premier League (IPL) season using historical IPL data. The project demonstrates the application of advanced machine learning techniques to sports data, with a focus on team and player performance metrics.

Objectives
1. Data Cleaning and Feature Engineering
2. Exploratory Data Analysis (EDA)
   1. Team Performance 
   2. Player Performance (Analyze team and player performance using historical IPL data)
   3. Seasonal Performance
3. Feature Extraction 
4. Building a predictive model to forecast the winner of IPL 2025.
   *. Experiment with ensemble models (e.g., Random Forest, XGBoost) and neural networks.

Dataset
The dataset used in this project includes:
1.Matches: Information about IPL matches from 2008 to 2024.(given)
2.Deliveries: Ball-by-ball data for each match.(given)
3.Team Performance: Aggregated statistics for each team.(Created as Part of EDA for model building)
4.Player Performance: Statistics for individual players.(Created as Part of EDA for model building)

1. Data Preprocessing:
     1. Handling Missing and Duplicated Values
     2. Standardize The inconsistency in team names
     3. Remove Unnecessary Columns
2. EDA
   1. Team Performance
      	Plot of Matches Played and Winning Percentages
	Plot of Run Rate and Economy Rate (as a bowling side)
	Plot of Highest and Lowest Scores
	Plot of Total 4s and 6s
	Plot of Average Powerplay and Death Overs Score
	Powerplay Analysis
   2. Player Performance
	Got the top 20 run-scorers
	Plot of Batting Average vs Batting Strike Rate for the top 20 run-scorers
	Highest Average and Strike Rate for players with >50 matches
	Plot of top wicket-takers
	Plot of top highest individual scores
	Man of the Match Count Analysis
	Used K-Means Clustering to plot Batting Average vs Bowling Economy Rate for number of clusters = 3 (Batsman, Bowler, All Rounder)
	Identified Top 10 Batsmen in each run category:
		Top 6’s scorer
		Top 4’s scorer
		Top 2’s scorer
		Top 1’s scorer
   3. Seasonal Analysis
	Calculate average runs per match per season
	Identify targets of 200+ runs per season
	Find the average score of each team per season
	Analyze runs of Orange Cap Holders per season
	Track wickets of Purple Cap Holders per season
	Find top 10 bowlers per season   


3. Feature Extraction:
     selected most useful features that most suits for our Model Training from the available 4 data frames

4. Model Creation
📊 Methodology

1. Data Preprocessing
  	Data Merging: The matches and deliveries datasets were merged using the match_id and id columns to create a comprehensive dataset.
 Feature Engineering:
   Categorical variables (team1, team2, venue, city, toss_winner, toss_decision) were encoded using LabelEncoder.
   Missing values were handled using SimpleImputer with a mean strategy.
   Target Encoding: The target variable (winner) was encoded using LabelEncoder to convert team names into numerical labels.

2. Model Development
   Random Forest: A Random Forest classifier with 100 estimators was trained on the training data.
   XGBoost: An XGBoost classifier with 100 estimators was trained on the training data.
   Neural Network: A Multi-Layer Perceptron (MLP) classifier was optimized using RandomizedSearchCV to find the best hyperparameters.
   Ensemble Model: A VotingClassifier was created by combining the Random Forest, XGBoost, and optimized Neural Network models. The ensemble used soft voting to predict the winner.

3. Model Evaluation
- The models were evaluated on a test set (20% of the data) using accuracy, precision, recall, and F1-score.
- The ensemble model's performance was compared with individual models to determine the best-performing approach.

4. Simulation for 2025 IPL Season
- Hypothetical match scenarios for the 2025 IPL season were created by iterating through all possible team combinations, venues, and toss decisions.
- The trained ensemble model was used to predict the winner for each simulated match.
- The most likely winner of the 2025 IPL season was determined by finding the mode of the predicted winners.

---

📈 Results

Model Performance
Random Forest:
  - Accuracy: 89.93%
  - Classification Report: High precision and recall across all classes, indicating strong performance.
XGBoost:
  - Accuracy: 89.97%
  - Classification Report: Comparable to Random Forest, with slightly better precision for some classes.
Neural Network:
  - Accuracy: 71.33%
  - Classification Report: Lower performance compared to tree-based models, likely due to the complexity of the data.
Ensemble Model:
  - Accuracy: 90.01%
  - Classification Report: The ensemble model outperformed individual models, achieving the highest accuracy and balanced precision/recall across all classes.
Predicted Winner of 2025 IPL Season
- The ensemble model predicted *Kolkata Knight Riders* as the most likely winner of the 2025 IPL season.

---

🎯 Conclusion

Key Findings
1. *Model Performance*:
   - The ensemble model combining Random Forest, XGBoost, and Neural Network achieved the highest accuracy (*90.01%*), demonstrating the effectiveness of ensemble methods for this task.
   - Tree-based models (Random Forest and XGBoost) outperformed the Neural Network, likely due to their ability to handle categorical data and non-linear relationships more effectively.

2. *Team Performance Insights*:
   - Teams with strong historical performance (e.g., Mumbai Indians, Chennai Super Kings) were consistently predicted to perform well in the 2025 season.
   - The toss decision and venue played significant roles in match outcomes, as reflected in the model's predictions.

3. *Predicted Winner*:
   - The model predicted *Kolkata Knight Riders* as the most likely winner of the 2025 IPL season, based on their simulated performance across various match scenarios.

### *Limitations*
1. *Data Constraints*:
   - The model relies on historical data, which may not account for unforeseen factors such as player injuries, team changes, or external conditions (e.g., weather).
2. *Neural Network Performance*:
   - The Neural Network underperformed compared to tree-based models, suggesting that more advanced architectures or hyperparameter tuning may be required.

### *Future Work*
1. *Feature Enhancement*:
   - Incorporate additional features such as player form, weather conditions, and team dynamics to improve model accuracy.
2. *Advanced Models*:
   - Experiment with deep learning models (e.g., LSTM, GRU) to capture temporal patterns in match data.
3. *Real-Time Predictions*:
   - Develop a real-time prediction system to update match outcomes dynamically based on live data.
