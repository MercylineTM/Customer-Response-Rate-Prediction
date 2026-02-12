# Customer-Response-Rate-Prediction
The purpose of this project is to develop a data-driven customer targeting solution for a superstore’s promotional campaign. The project focuses on predicting customer response, identifying key behavioral drivers, and presenting insights through an interactive decision dashboard.
The scope includes:
•	Analysis of existing customer data
•	Predictive modeling to estimate response likelihood
•	Visualization and reporting to support marketing decisions

Project Objectives
•	Predict the likelihood of customers accepting a Gold Membership promotional offer
•	Reduce campaign costs by optimizing call targeting
•	Identify customer behaviors that influence acceptance
•	Segment customers into high, medium, and low response probability groups
•	Provide actionable insights via an interactive dashboard

Problem Statement
Traditional marketing campaigns often rely on blanket outreach, which is costly and inefficient. In this case, the superstore planned to promote a Gold Membership offer via phone calls to existing customers.
Key challenge:
How can the company identify which customers are most likely to accept the offer, instead of calling everyone?
Without predictive targeting:
•	Campaign costs increase
•	Conversion rates remain low
•	Customer experience may deteriorate
This project aims to solve this inefficiency using data analytics and predictive modeling.

 Key Performance Indicators (KPIs)
•	Response Rate
•	Predicted Response Probability
•	Precision & Recall
•	ROC–AUC
•	Lift & Gains
•	Target Coverage (% of customers contacted)

Tools & Technologies
•	Python
•	Pandas, NumPy
•	Scikit-learn
•	Matplotlib
•	Streamlit
•	CSS (UI customization)

Data Description
The dataset contains demographic, behavioral, and transactional data for existing customers who were previously targeted in marketing campaigns.
The target variable indicates whether a customer accepted a past campaign offer.

Data Cleaning Steps
•	Missing Values
o	Income contained missing values
o	Imputed using median to avoid skew
•	Duplicates
o	Checked using customer ID
o	No duplicate records found
•	Outliers
o	High-spend values reviewed
o	Retained as they represent valid premium customers

Feature Engineering:
o	Age = Current Year – Year_Birth
o	Customer Tenure = Years since enrollment
o	Total Spend = Sum of product category spend
o	Total Purchases = Sum of all purchase channels
•	Categorical Encoding:
o	One-hot encoding applied to Education and Marital Status
•	Scaling:
o	Numerical features standardized for modeling


