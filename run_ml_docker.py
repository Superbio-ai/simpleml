import argparse
from ml_functions import check_inputs, run_simple_pipelines
import pandas as pd


def populate_predictors(df, args):
    if len(args.predictors) == 0:
        predictors = list(df.columns)
        predictors.remove(args.target)
        args.predictors = predictors
    else:
        #string to list, with elements separated by ,
        args.predictors = args.predictors.split(',')
    
    return(args)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", help="name of file uploaded to container, which contains data", type=str, required=True, default='test_data.csv'
    )
    parser.add_argument(
        "-pt", "--problem_type", help="regression or classification", type=str, required=True, default='regression'
    )
    parser.add_argument(
        "-mt", "--model_type", help="linear_model, random_forest, gradient_boosting, neural_network, or best", type=str, required=True, default='random_forest'
    )
    parser.add_argument(
        "-p", "--predictors", help="columns in dataset to use as predictors, provided as a list separated by commas. If empty will be all columns except for target", type=str, required=True, default=''
    )
    parser.add_argument(
        "-t", "--target", help="column in dataset to use as target", type=str, required=True, default='Target'
    )
    parser.add_argument(
        "-r", "--train_test_ratio", help="ratio of input data to use for training. rest of data is heldback for testing", type=float, required=True, default=0.7
    )
    parser.add_argument(
        "-m", "--missing_treatment", help="either drop rows with missing data, or replace missing data with mean of each column", type=str, required=True, default='drop'
    )
    args = parser.parse_args()
    
    df_in = pd.read_csv(args.file)
    
    args = populate_predictors(df_in, args)
    
    check_inputs(args.problem_type, args.model_type, args.predictors, args.target, args.train_test_ratio, args.missing_treatment, df_in)
    
    means = run_simple_pipelines(df_in, args.problem_type, args.model_type, args.predictors, args.target, args.train_test_ratio, folds = 5, missing_treatment = args.missing_treatment)
    
    '''
    df = pd.read_csv(args.file)

    #predictors = list(df.columns)[6:]
    
    check_inputs(problem_type, model_type, predictors, target, train_test_ratio, missing_treatment, df)
    means = run_simple_pipelines(df, problem_type, model_type, predictors, target, train_test_ratio, folds = 5, missing_treatment = missing_treatment)
    '''
    means.to_csv('results.csv')
    