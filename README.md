# Data Science Salary Estimator based on Glassdoor data: Project Overview
* Built a tool that can help estimating a data scientist salary (helpful for me to know what to expect when landing my first job as DS)
* The data was scraped from Glassdoor using Python/Selenium (USA data). Data from other countries can be used if Glassdoor provid that
* Created additional features from the job description of each job (e.g., python, spark, excel, aws)
* Applying different regression models (linear regression, lasso regression, reandom forest) to the data to find the best model.
* Built a client facing API using flask 

## Programming Language & Packages
* Python
* Packages: numpy, pandas, matplotlib, seanborn, statsmodel, sklearn, selenium, pickle, flask

## Resources
* For Scraping: https://github.com/arapfaik/scraping-glassdoor-selenium 
* Flask Productionization: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Data scraped
When scraping the glassdoor.com for data scientist jobs, the following data was collected for each posted job:
* Job title
* Salary Estimate
* Job Description
* Rating
* Company
* Location
* Company Headquarters
* Company Size
* Company Founded Date
* Type of Ownership
* Industry
* Sector
* Revenue
* Competitors

## Data Cleaning
Before applying the data to the Machine Learning models, some data_cleaning was applied to the data and, some extra features was added to the data.
* Removed rows without salary
* Parsed rating out of company text
* Made a new column for company state
* Parsed numeric data out of salary
* Made columns for employer provided salary and hourly wages
* Added a column for if the job was at the company’s headquarters
* Transformed founded date into age of company
* Column for simplified job title and Seniority
* Made columns for if different skills were listed in the job description:
  * Python
  * R
  * Excel
  * AWS
  * Spark
* Column for description length

## Exploratory Data Analysis (EDA)
Checked the distributions and built many pivot tables to answer some questions related to the salray (e.g., salary based on state, salary based on experience, salary based on skills)
I also plot a heatmap for some of the features to check the correlations between them.
