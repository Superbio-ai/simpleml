from flask import request
import pandas as pd
from ml_functions import check_inputs, run_simple_pipelines

from flask_restful import Resource
import logging
from logging.handlers import WatchedFileHandler

import json


class WrapResource(Resource):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.log = 'application.log'

    @classmethod
    def set_logging(cls, log_file: str):
        handler = WatchedFileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s  [%(levelname)s]\n%(message)s",
            "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel("INFO")
        root.addHandler(handler)


def _toJSON(object):
        return json.dumps(object, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
    
    
class CrossValidate(WrapResource):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def post(self):
        try:
            #get info
            problem_type = request.form.get('problem_type')
            model_type = request.form.get('model_type')
            predictors = eval(request.form.get('predictors'))
            target = request.form.get('target')
            train_test_ratio = float(request.form.get('train_test_ratio'))
            missing_treatment = request.form.get('missing_treatment')
            file = list(request.files.values())[0]
            data = file.stream
            df_in = pd.read_csv(data)
            
            check_inputs(problem_type, model_type, predictors, target, train_test_ratio, missing_treatment, df_in)
            
            means = run_simple_pipelines(df_in, problem_type, model_type, predictors, target, train_test_ratio, folds = 5, missing_treatment = missing_treatment)
            
            json_response = {'fit_time': means['fit_time'], 'score_time': means['score_time'],
                        'test_score': means['test_score'], 'train_score': means['train_score']}
            
            return json_response, 200

        except ValueError as e:
            print(e)
            return {"error": _toJSON(e)}, 400
        except Exception as e:
            return {"error": str(e)}, 500