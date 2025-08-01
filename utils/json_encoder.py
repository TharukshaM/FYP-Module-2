import json
import numpy as np
import pandas as pd
from flask.json.provider import DefaultJSONProvider

class CustomJSONEncoder(DefaultJSONProvider):
    """Custom JSON encoder that handles NumPy and Pandas types"""
    
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, default=self.json_serializer, **kwargs)
    
    def loads(self, s):
        return json.loads(s)
    
    @staticmethod
    def json_serializer(obj):
        """JSON serializer for objects not serializable by default json code"""
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # Handle numpy arrays
            return obj.tolist()
        
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj