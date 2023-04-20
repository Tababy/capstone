# Model Card

See the [example Google model cards](https://modelcards.withgoogle.com/model-reports) for inspiration. 

## Model Description

**Input:** Describe the inputs of your model 
The inputs of the model are Tehcnical indicators, Strategies for technical indicators, and Weighted sums for those technical indicators. Also, I purpose adding two columns created using a simple kmeans model, using k=2 and k=3 groups ( This were purposed by elbow method).

**Output:** Describe the output(s) of your model
Either 1, 0 or -1 (3 possible values, meaning BUY, NEUTRAL and SELL respectively)
**Model Architecture:** Describe the model architecture you’ve used
I created first a simple kmeans algorithm for enriching features. I then created a simple Random Forest Classifier as Baseline model. 
I Finally created a XGB model, with parameters optimized by Bayesian Optimization. The hyperparameters to be tuned were:

search_space = {
    'learning_rate': (0.01, 0.5),
    'max_depth': (3, 10),
    'n_estimators': (50, 500),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'gamma': (0, 1),
    'colsample_bytree': (0.1, 1),
}



## Performance

Give a summary graph or metrics of how the model performs. Remember to include how you are measuring the performance and what data you analysed it on.

Test model accuracy: 0.7962962962962963
Test model precision: 0.78930362654321
Test model recall: 0.7962962962962963
Test model F1-score: 0.79275547799108

## Limitations


There are several limitations of the model. 
- First of all, the Combined_Strategy_target can be imporved to look for better moments to invest than the ones purposed (using local minima and maxima search)
- Also, the weight given to the periods of time is somehow arbitrary, I think ML could also be used to get the "best weights", although, another target variable should be created and the indicator will be treated merely as another dimension
- There are a lot more models that can be used (including NN)

- For the present work, is a good Proof of Concept on the usage of raw data to predictions.

## Trade-offs

Outline any trade-offs of your model, such as any circumstances where the model exhibits performance issues. 

Mainly, for this preliminary apporach, I only take the last 1440 minutes.recordsV (the whole day for April 4 2023).
