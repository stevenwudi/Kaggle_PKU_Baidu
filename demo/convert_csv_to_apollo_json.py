import pandas as pd
import os
import json
from tqdm import tqdm

csv_dir = 'E:\DATASET\pku-autonomous-driving\submissions'

csv_name = 'Dec21-21-32-32_epoch_45_conf_0.1'
csv_file = os.path.join(csv_dir, csv_name + '.csv')
train_df = pd.read_csv(csv_file)

output_dir = os.path.join(csv_dir, csv_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for i in tqdm(range(len(train_df))):
    json_name = train_df.iloc[i]['ImageId'] + '.json'
    if not type(train_df.iloc[i]['PredictionString']) == float:  # not prediction
        prediction = train_df.iloc[i]['PredictionString'].split(' ')

        cars = []
        for c_idx in range(int(len(prediction)/7)):
            car = {}
            car["car_id"] = 0   ### We need to modify this later
            car["pose"] = [float(x) for x in prediction[c_idx*7: c_idx*7 + 6]]
            cars.append(car)

        outfile = os.path.join(output_dir, json_name)
        with open(outfile, 'w') as f:
            json.dump(cars, f, indent=4)
