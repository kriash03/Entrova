Entrova is an AI-driven fraud detection system that leverages machine learning and blockchain to ensure secure and transparent transactions. The project integrates real-time data processing with AI analytics to detect fraudulent behavior in financial transactions.

After downloading the app or copying the repo 

create a virtual environment by-> python -m venv "name of the virtual env"
then run it by-> source "name of the virtual env"/bin/activate
(both of the above are to be run on terminal)

then 
run (in the terminal)-> pip install pandas numpy vaderSentiment scikit-learn imbalanced-learn streamlit plotly-express 

check with the requirement.txt regarding version compatibility

after doing so 
delete all the csv files EXCEPT "sample_influencer_database_large.csv"

then go to the ipynb file and run the cells separately 

after all the cell processes are completed 

run (in the terminal)-> streamlit run app.py
