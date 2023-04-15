# Prediction-with-Pycaret

# Objective 
To attempt to forecast Europe Air BnB rental rates for both business and leisure travelers in various European cities using pycaret and replicating its results in sklearn.

# Technologies Used

* [Pycaret](https://pycaret.org/)
* [skLearn](https://scikit-learn.org/stable/index.html)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://scikit-learn.org/stable/index.html)
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)

# Pycaret
Pycaret is a low code python library used for data preparation, model training, hyperparameter tuning, model evaluation, and deployment. 

## Data preparation and setup
Data preparation and setup

The type of model choose was regression. Because of the effect of outliers on the predictions it was decided in an attempt to get a better model that the outliers be removed. However, the data was reserved and exported to csv for future use if needed.

The setup then does all the heavy lifting:
1) One hot encoding
2) Imputations
3) Train and Test split

With many more options for customizations depending on the dataset.

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/Setup%20Options.gif" align="center">

## Compare model
Pycaret runs compares a total of 20 models. Giving the user options for best models to test. Though as said by British statistician George E. P. Box "All models are wrong, but some are useful." 

Compare models gives a good place to start to find those useful models for futher testing
<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/CompareModels.png" align="center">

## Create, tune and finalize model
Once a model is choosen it is then created, tuned and finalized. 
```
xgb = create_model('xgboost')
t_xgb= tune_model(xgb)
f_xgb = finalize_model(t_xgb)
f_xgb
```
# Visualization
Results from model then observed using the following charts

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/FeatureImportance.png" align="center">

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/ResidualChart.png" align="center">

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/PredictionError.png" align="center">

# Predicing the model
Finally the model is tested against unseen data to see how it perform
```
pred = predict_model(f_xgb,data=data_unseen)
pred = pred.loc[:,['cost per night cad','prediction_label']]
df_pred['Percent Diff'] = (df_pred['cost per night cad']-df_pred['prediction_label'])/df_pred['cost per night cad']
df_pred

df_pred.describe().T
```
<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/HyperParameters.png" align="center">

# Click to see skLearn version of predictions
The result of tuning the model gives us hyper parameters that we can now take to skLearn for finer control the learning process of the model. Iteration is key!

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/HyperParameters.png" align="center">

# References
Barroso, G. (2018, May 3). Admore ITN. AdMoRe ITN. Retrieved April 14, 2023, from https://www.lacan.upc.edu/admoreWeb/2018/05/all-models-are-wrong-but-some-are-useful-george-e-p-box/ 
