# predict-escape-fraction
Predicting the effects of SARS-Cov-2 spike protein RBD mutations on immune escape from neutralizing antibodies

1. File 'predictions.py' contains the code for machine learning model at 30% threshold, model performance, 10-fold cross validation performance and the predicted results.
2. The input file for predictions.py is 'Dataset1.csv' present in Dataset folder.
3. The input file for making the predictions is 'data_for_prediction.csv'.
4. File 'threshold model.py' contains the code for model performance at different selection threshold from 5-30%. The input file is 'Dataset1.csv'. This program prints the performance of the model at different thresholds.
5. The dataset folder contains the input data and the output predictions from the model.
6. The predicted results are saved in the file named 'final_predictions.csv'.
