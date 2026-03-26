import numpy as np
import pandas as pd


def test_param_recovery_results(results: pd.DataFrame):
    assert isinstance(results, pd.DataFrame), "❌ results should be a DataFrame"
    assert len(results.columns) == 4, "❌ the `results` DataFrame should have 4 columns"
    expected_column_names = ['learning_rate', 'temperature', 'recovered_learning_rate', 'recovered_temperature']
    for column_name in expected_column_names:
        assert column_name in results.columns, f"❌ the `results` DataFrame should have a column named {column_name}"
    assert np.all(results > 0), "❌ all values in the `results` DataFrame should be positive"
    print("✅ OK 👌")

