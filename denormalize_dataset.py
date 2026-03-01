import pickle
import copy

def denormalize(data):
    with open('data/normalization_params.pkl', 'rb') as f:
        normalization_params = pickle.load(f)
    x_mean = normalization_params['x_mean']
    x_std = normalization_params['x_std']
    y_mean = normalization_params['y_mean']
    y_std = normalization_params['y_std']

    print(f"Global X Mean: {x_mean}, X Std: {x_std}")
    print(f"Global Y Mean: {y_mean}, Y Std: {y_std}")
    denormed_data = []
    for entry in data:
        denormed_entry = copy.deepcopy(entry)
        denormed_XY = []
        for row in entry["X_Y_combined"]:
            x1 = row[0] * x_std + x_mean
            x2 = row[1] * x_std + x_mean
            y = row[2] * y_std + y_mean
            denormed_XY.append([x1, x2, y])
        denormed_entry["X_Y_combined"] = denormed_XY
        denormed_data.append(denormed_entry)
    return denormed_data