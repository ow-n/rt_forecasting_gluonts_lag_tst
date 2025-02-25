# Context
Welcome back 😀
So the model trains and tests off of a train.csv and test.csv file
I wrote a data_preprocess.py file that generates those csv files based off our Oklahoma dataset

So change the data_preprocess.py file if we need to change how we train the model (selecting file range)

After testing the model, it'll generate a 'predictions.csv' which is just the predicted values against real values (test data)
    So I wrote the src_owen/evaluate.py file to graph the comparison


## Options:
1) Reconfigure training/test files:
    > python src_owen/data_preprocess_oklahoma.py
    > python data_preprocess_synthetic.py
2) Train
    > python src/train.py
3) Predict
    > python src/predict.py
4) Evaluate Predictions:
    > python src_owen/evaluate.py

## Project Structure Context
Dataset:
- 1 minute per file
- first 2 files is normal
- freq is 33ms

Test.csv has 100 samples instead of length of training
- based on the forecast length
    - 100 means we're only going to predict 100 samples, all at once
- so there can only be 100 samples to compare to
- has no effect on actual predicted value
    - model uses the stored training data to predict the next values instead

Forecast Length x Ratio x Lag
- Forecast length is how many samples we're going to predict
- Ratio calculates the History Window to use from Training data
    - Ratio=15, -> HistoryWindow=1500, -> last 1500 samples from training used to predict
- LagWindow is smaller window that determines input features


# Configs
> model_input_outputs/schema/schema.json
    - adjust forecast length
    - decides what columns to use in train/test csv
> src/config/default_hyperparameters.json

