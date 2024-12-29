from pathlib import Path
import os

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_DIR = os.path.join(ROOT_DIR, "models")
    MODEL_NAME = "model.joblib"
    SCALER_NAME = "scaler.joblib"
    
    FEATURES = [
        'Pclass',
        'Age',
        'Fare',
        'Sex_female',
        'Sex_male',
        'Embarked_C',
        'Embarked_Q',
        'Embarked_S'
    ]

