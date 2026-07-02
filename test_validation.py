import pandas as pd

def validate_dataset(df):
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."
    assert df.duplicated().sum() == 0, "Dataset contains duplicate rows."

def test_sample_dataset():
    data = {
        "Response Tat Bd": [1.0, 2.0, 3.0],
        "Business Segment": ["A", "B", "C"]
    }

    df = pd.DataFrame(data)

    validate_dataset(df)

    print("Validation test passed.")

if __name__ == "__main__":
    test_sample_dataset()
