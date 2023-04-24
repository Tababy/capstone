**DOWNLOAD DATASET CREATED FOR THE CAPSTONE**

https://www.kaggle.com/datasets/tababy/btcdatawithtechnicalindicatorsforallperiods2023

(Same dataset can be reproduced usingscripts in ETL)
    

# A RAW Proof of Concept of a FULL DataEngineering/DataScience Pipeline


Eventhough I dont stand the pace of the competition, and because of that, I was not able to document therefore my evolution on the project, development, I want to show that I fullfill the objective of the capstone, which were:

    - Optimise an existing or original code base to run against a test function.
    - Reflect on the application of machine learning and artificial intelligence to career goals.
    - Apply Bayesian optimisation to a real-life data challenge in a specific industry.

In this project, BTC data from binance was scrapped, cleaned and then Technical inidcators were added for each period of time. Then, a big tbale with all periods of time was created. After that, strategy columns were created for each technical inidcator for each period of time. Then, after finishing the ETL process, a technical indicator was created using a weighted sum of all the strategies in all periods of time. 

## DATA

                                            ETL Process:
The periods of time are:
    [min,5min,10min,15min,30min,hour,2hour,4hour,8hour,day,week]
For example, the Technical indicator RSI_14 (Relative Strength Index takng 14 periods of time) will have columns as:
    - RSI_14_{period} for each of the periods
    - Strategy_RSI_14_{period} for each of the periods
    - It has also some weighted columns, which takes all weights for all periods of time the same (as 1, more below on weights)

This whole process create the Kaggle Dataset upload, all the notebooks and scripts for the process were added completition (in the folder ETL, which has the STREMING ---> PREPROCESSING ---> ANALYSIS), but are not necesarry to be runed by instructors to hop en the capstone.ipynb (just to download the kaggle dataset I upload, and have it in the same folder as the capstone.ipynb)

NOTES: 
    - BY NO MEANS THE CODE PROVIDED IN ETL IS GENERALIZED, STANDARIZED AND OPTIMIZED, BUT IT SERVES FOR A PRELIMINARY PIPELINE TO CREATE THE BTC KAGGLE DATASET FROM SCRATCH USING BINANCE DATA.

    - IN THE ANALYSIS FOLDER, THERE IS A NOTEBOOK WITH CUSTOM VOICE ALARMS FOR CERTAING INDICATORS AND CHANGES WITH MORE THAN +100 VARIATIONS I CREATED, VERY SUITABLE FOR HIGH LEVARAGE TRADES IN WHICH YOU HAVE TO HAVE AN EYE VERY CLOSE TO THE INDICATORS (NOW YOU CAN DO OTHER STUFF AND THE ALARMS WILL STREAM DATA EVERY MINUTE AND TELL YOU LITERALLY CHANGES IN CERTAIN INDICATORS OR PRICE ITSELF OFR DIFFERENT PERIODS OF TIME)


                                            Data Science Pipeline
For this, I do a basic sanity check, eliminating columns that arent suitable for the analysis (closing times, which are redundant because we have the open_time as primary key) and also we Delete Fibonacci Retracements (a technical indicator) , which for some reason had inf values and will be debugged in future versions. 

Then, for modeling, a target variable should be created. For this, a path is to get the "best times to open a BUY/SELL position"  (note that the best time to close a BUY position is the best time to open a SELL posiiton and viceversa). This makes this a min/max problem in which we could find the best dates to invest investing in peaks in the graphs; this idea will be implemented in future works. But if we do this, using tresholds of 10% changes (for both upPrices and downPrices), we get only around 100 registers out of the +1000000 rows of the dataset as good moments (so for this, a model that is good for identifying  sparse moemnts may be good, or getting some oversmaling techniques)

In this work, instead, I created weighted sums of each indicator. (weigted_RSI_14 wold be the wieghted sum of the RSI indicator). I created a "Combined_Strategy", which is a sum of all weighted indicators. Then, I define a treshold for both, the buy and sell position using the outliers of the data. THis creates the "Combined_Strategy_target" 9which takes values 1,0,-1 for BUY,NEUTRAL,SELL repsectively)column, which we will use as a target variable in the models. We give more weight to indicators in higher periods of time (as for 1 minute graphs, BUY/SELl signals happen more ofthen than in 4 hour graphs).

## MODEL 

Some feature extraction was done using RFE(Recursive Feature Elimination) and Random Forest Classifier, to leave a total of 10 features. 

## HYPERPARAMETER OPTIMSATION

After that, data was divided in train, val and test split, with a ratio of 70/15/15. I created a baseline model using a simple random forest classifier with default parameters. And after that, train a base XGBoost model using Bayesian Optimization for the hyperparameter tuning. 

## RESULTS

At the end, I compare the results with the baseline model. 


It is important to note that I just take a subset of the data (150000 of the most recent registers/minutes) to do the feature extraction and to run the models. This was mainly for the proof of concept and demonstrating sufficceny at the degree of instructors consideration. 



## CONTACT DETAILS
LinkedIn: https://www.linkedin.com/in/izotsogp/

