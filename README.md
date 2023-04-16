<h1 align="center" style="color:MediumSeaGreen;"> <b>  Prediction-with-Pycaret</b>

# Objective 
The goal is to utilize pycaret to predict Air BnB rental prices in multiple European cities for both business and leisure purposes.
# Technologies Used

* [Pycaret](https://pycaret.org/)
* [pandas](https://pandas.pydata.org/)

# Pycaret
Pycaret is a low code python library used for data preparation, model training, hyperparameter tuning, model evaluation, and deployment. 

## Data preparation and setup the model
A regression model was selected, but due to the impact of outliers on the accuracy of the predictions, it was deemed necessary to eliminate them in order to achieve a more effective model. However, the data was preserved and exported to a CSV file for potential future use.

The subsequent steps of the process handled the bulk of the work:

1) Conducting one-hot encoding
2) Implementing imputations
3) Separating data into training and testing sets
  
Additional adjustments can be made depending on the specific dataset.
  
<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/Setup%20Options.gif" align="center">

## Compare the model
Pycaret executes and evaluates 20 different models, allowing the user to select the most promising models for further examination. However, as British statistician George E. P. Box famously stated, <b>"All models are wrong, but some are useful."</b>

Comparing the models provides a solid foundation for identifying potentially valuable models to explore more extensively.
<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/CompareModels.png" align="center">

## Create, tune and finalize the model
After a model has been selected, it is developed, fine-tuned, and ultimately completed. In this instance, the xbgboost model was chosen based on its strong showing in the comparison process, in top 5, and personal familiarity with it.
```
xgb = create_model('xgboost')
t_xgb= tune_model(xgb)
f_xgb = finalize_model(t_xgb)
f_xgb
```
# Visualization
Results from model then observed using the following charts.

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/FeatureImportance.png" align="center">

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/ResidualChart.png" align="center">

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/PredictionError.png" align="center">

# Predict the model
Finally the model is tested against unseen data to see how it perform
```
pred = predict_model(f_xgb,data=data_unseen)
pred = pred.loc[:,['cost per night cad','prediction_label']]
df_pred['Percent Diff'] = (df_pred['cost per night cad']-df_pred['prediction_label'])/df_pred['cost per night cad']
df_pred

df_pred.describe().T
```
<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/DescriptiveTable.png" align="center">
  
Upon examining the outcome, it was found that the mean deviation between the projected and actual cost, in comparison to unobserved data, was -3.29%.

# [Click to see SKLearn version of predictions](https://github.com/Piettro314/Prediction-with-skLearn)

Fine-tuning the model yields hyperparameters that can be utilized in skLearn to exercise greater control over the model's learning process. Persistence in this regard is crucial, as it involves an iterative process.

<a href="https://github.com/Piettro314/Prediction-with-skLearn"><img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/HyperParameters.png" align="center" /></a>

# References
Barroso, G. (2018, May 3). Admore ITN. AdMoRe ITN. Retrieved April 14, 2023, from https://www.lacan.upc.edu/admoreWeb/2018/05/all-models-are-wrong-but-some-are-useful-george-e-p-box/ 
