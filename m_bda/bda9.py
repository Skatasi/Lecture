from matplotlib import pyplot as plt
import pandas as pd

data = {'Name':['Ada', 'Alex', 'Bill', 'Daisy', 'David', 'John', 'Kate', 'Lewis', 'Lily', 'Mary'],
        'Gender':[1,0,0,1,0,0,1,2,1,1], 
        'Age':[39, 53, 25, 28, 45, 37, 48, 40, 52, 43], 
        'Position':[0,1,0,0,1,0,1,0,1,0], 
        'BMI':['?', 31.7, 20.5, 22.7, 30.2, 21.6, 26.3, 24.4, 28.0, 28.6], 
        'Drink':[1,2,0,1,2,0,2,2,2,0], 
        'Smoke':[0,1,0,0,1,1,1,0,0,0], 
        'Total_Cholesterol':[172,255,159,166,242,180,181,192,201,215]}

import numpy as np
import pandas as pd

# Prepare the data
df = pd.DataFrame(data)
ada_data = df[df['Name'] == 'Ada']
df = df[df['BMI'] != '?']
df['BMI'] = df['BMI'].astype(float)
print(ada_data)
# Find the k-nearest neighbors of Ada in the same gender class, where k = 2

same_gender_class = df[df['Gender'] == ada_data['Gender'][0]]
same_gender_class = same_gender_class[same_gender_class['Name'] != 'Ada']

# Calculate the distance by the difference in total cholesterol
same_gender_class['Distance'] = abs(same_gender_class['Total_Cholesterol'] - ada_data['Total_Cholesterol'][0])
print(same_gender_class)

# Find the 2 nearest neighbors
k = 2
nearest_neighbors = same_gender_class.nsmallest(k, 'Distance')
print(nearest_neighbors)

# Compute Ada's BMI by a weighted average over the k-nearest neighbors
weights = 1 / nearest_neighbors['Distance']
weights = weights / weights.sum()  # Normalize weights to sum up to 1
ada_bmi = np.sum(weights * nearest_neighbors['BMI'])

print(f"Predicted BMI for Ada: {ada_bmi}")
