import pandas as pd
from types import SimpleNamespace

from shapoint_webapp.model_manager import ModelManager


def test_rule_based_scoring():
    # Minimal config for ModelManager
    cfg = {
        'model': {
            'task': 'Classification',
            'k_variables': 1,
            'max_leaves': 2,
            'score_scale': 10,
            'use_optuna': False,
            'n_random_features': 0,
            'params_path': '',
            'model_path': ''
        },
        'data': {
            'default_dataset': 'breast_cancer',
            'data_path': '',
            'data_separator': ',',
            'target_column': 'target',
            'exclude_columns': [],
            'feature_types': {},
        }
    }

    mm = ModelManager(cfg)

    # Stub model with summary containing rule for feature 'age'
    summary_dict = {
        'feature_summary': {
            'age': {
                'levels_detail': [
                    {'rule': 'age >= 50', 'scaled_score': 2},
                    {'rule': 'age < 50', 'scaled_score': 0},
                ]
            }
        }
    }

    class StubModel:
        def get_model_summary_with_nulls(self):
            return summary_dict

    mm.model = StubModel()
    mm.feature_names = ['age']
    mm.feature_types = {'age': 'continuous'}

    # Build input dataframe with age 60
    df = pd.DataFrame([{'age': 60}])

    contribs = mm._get_feature_contributions(df)
    assert contribs[0]['score'] == 2
    assert contribs[0]['contribution'] == 2 