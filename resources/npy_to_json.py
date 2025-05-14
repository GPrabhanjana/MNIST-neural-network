import numpy as np
import json

# Load the .npy file
data = np.load('mnist_model_params.npy', allow_pickle=True).item()

# Create a custom encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                             np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# Save to JSON file
with open('model_weights.json', 'w') as f:
    json.dump(data, f, cls=NumpyEncoder)

print("Conversion complete! Saved to model_weights.json")