import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('influencer_matching_scores.csv')

# Simulate the target variable based on a lower threshold
df['Success'] = df['Matching_Score'].apply(lambda x: 1 if x >= 50 else 0)

# Features (X) and target variable (y)
X = df[['Engagement_Rate', 'Audience_Match_Score', 'Content_Match_Score', 'Sentiment_Score', 'Adjusted_Fraud_Score']]
y = df['Success']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Random Oversampling to balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)

# Streamlit app
st.title("Influencer-Brand Matching Tool")

# Sidebar for filters and campaign goals
st.sidebar.header("Filters and Campaign Goals")

# Filters for niche and platform
niche_filter = st.sidebar.selectbox("Filter by Niche", df['Niche'].unique())
platform_filter = st.sidebar.selectbox("Filter by Platform", df['Platform'].unique())

# Input fields for campaign goals
st.sidebar.subheader("Campaign Goals")
engagement_rate = st.sidebar.slider("Engagement Rate", min_value=0.0, max_value=10.0, value=5.0)
audience_match_score = st.sidebar.slider("Audience Match Score", min_value=0.0, max_value=100.0, value=50.0)
content_match_score = st.sidebar.slider("Content Match Score", min_value=0.0, max_value=100.0, value=50.0)
sentiment_score = st.sidebar.slider("Sentiment Score", min_value=0.0, max_value=100.0, value=50.0)
adjusted_fraud_score = st.sidebar.slider("Adjusted Fraud Score", min_value=0.0, max_value=100.0, value=50.0)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Engagement_Rate': [engagement_rate],
    'Audience_Match_Score': [audience_match_score],
    'Content_Match_Score': [content_match_score],
    'Sentiment_Score': [sentiment_score],
    'Adjusted_Fraud_Score': [adjusted_fraud_score]
})

# Display the input data for debugging
st.write("Input Data for Prediction:")
st.write(input_data)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the probability of success for the input campaign goals
matching_score = model.predict_proba(input_data_scaled)[0][1] * 100  # Probability of success
st.write(f"### Matching Score: {matching_score:.2f}%")

# Recalculate Matching Scores for all influencers based on the input campaign goals
df['Dynamic_Matching_Score'] = model.predict_proba(scaler.transform(X))[:, 1] * 100

# Display the recalculated matching scores for debugging
st.write("Recalculated Matching Scores:")
st.write(df[['Influencer_ID', 'Name', 'Dynamic_Matching_Score']])

# Filter all influencers by niche and platform
filtered_influencers = df[
    (df['Niche'] == niche_filter) & 
    (df['Platform'] == platform_filter)
]

# Display the filtered influencers before sorting
st.write("Filtered Influencers (Before Sorting):")
st.write(filtered_influencers[['Influencer_ID', 'Name', 'Dynamic_Matching_Score', 'Niche', 'Platform']])

# Sort the filtered influencers by Dynamic_Matching_Score
filtered_influencers = filtered_influencers.sort_values(by='Dynamic_Matching_Score', ascending=False)

# Display filtered influencers
st.write(f"### Influencers in {niche_filter} ({platform_filter}):")
if not filtered_influencers.empty:
    st.write(filtered_influencers[['Influencer_ID', 'Name', 'Dynamic_Matching_Score', 'Niche', 'Platform']])
else:
    st.write("No influencers found for the selected filters.")

# Visualizations
st.write("---")
st.write("### Influencer Distribution by Niche")

# Create a bar chart for influencer distribution by niche
niche_distribution = df['Niche'].value_counts().reset_index()
niche_distribution.columns = ['Niche', 'Count']

fig = px.bar(niche_distribution, x='Niche', y='Count', title="Influencer Distribution by Niche")
st.plotly_chart(fig)