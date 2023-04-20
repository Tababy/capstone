# Datasheet Template

As far as you can, complete the model datasheet. If you have got the data from the internet, you may not have all the information you need, but make sure you include all the information you do have. 

## Motivation

- For what purpose was the dataset created? 
For discerning good moments to open BUY/SELL Positions based on characteristics in high dinsional datapoints.
- Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset?
The Binance Team is resposible to gather the data daily in a csv file for that day, with 12 columns that have the tradiitonal features such as open, close, high and low prices. Also, they have data for the volume and buyer takes, aorund 12 dimensinal data. 

I then do a full ETL process, enriching the data to 925 dimensions
I then upload that to a kaggle dataset, and use it to proof of concept many aspect of topics seen along the course (including at the end Bayesian optimization for tuning the model)

 
## Composition

- What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? 
Historicla Stock Data and calculated technical indicators and strategies.
- How many instances of each type are there? 
1 dataset
- Is there any missing data?
Just for Fibonacci REtracements, which were eliminated in the Preprocessing for Data Science Pipeline stage
- Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by    doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?
No
## Collection process

- How was the data acquired? 
Download from Binance, and enriched after several days of hard work
- If the data is a sample of a larger subset, what was the sampling strategy? 
no, but for modelling, I started the first attempt with only the data from the last 1440 records (APril 4 2023 data)

## Preprocessing/cleaning/labelling

- Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section. 
exstensively, a lot of preprocessing to add technical indicators, strategies and target variables. Full details in here:
https://www.kaggle.com/datasets/tababy/btcdatawithtechnicalindicatorsforallperiods2023
- Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? 
Yes, in the ETL process scripts, all the zip files for each individual day are saves as they come raw form binance (although, for this work, only the Data Science pipeline is shared and the dataset is avaialble to download already form my Kaggle, as running all the ETL process will take to long for instructors)
## Uses

- What other tasks could the dataset be used for? 
Generl prupsose of scraping and analzing virtually any stock with minimum tweaks



- Who maintains the dataset?
Me
