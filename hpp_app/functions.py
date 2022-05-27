import pickle
import json
import numpy as np
import config


def get_predicted_house_price(total_sqft,bath,bhk,loc):
    loc = loc.lower()
    with open(config.MODEL_FILE, 'rb') as f:
        linear_model = pickle.load(f) 

    column_list = get_data_columns()

    x_user_data = np.zeros(len(column_list))

    column_list = list(column_list)
    print('column_list ::',column_list)
    loc_index = column_list.index(loc)
    x_user_data[0] = total_sqft
    x_user_data[1] = bath
    x_user_data[2] = bhk
    x_user_data[loc_index] = 1

    y_pred = linear_model.predict([x_user_data])
    predicted_price = y_pred[0]

    print("Predicted price is :",predicted_price)
    return round(predicted_price,2)

def get_location_names():
    print(config.COLUMN_NAMES_JSON_PATH)
    with open(config.COLUMN_NAMES_JSON_PATH, 'r') as f: 
        data_columns_dict = json.load(f) # dict 
        data_columns = data_columns_dict['Columns'] 
        locations = data_columns[3:]  
    return locations 


def get_location_names1():
    print(config.COLUMN_NAMES_JSON_PATH)
    with open(config.COLUMN_NAMES_JSON_PATH, 'r') as f: 
        data_columns_dict = json.load(f) # dict 
        data_columns = data_columns_dict['Columns'] 
        locations = data_columns[3:]  
    return locations 

def get_data_columns():
    with open(config.COLUMN_NAMES_JSON_PATH, "r") as f:
        data_columns = json.load(f)['Columns']
    return data_columns
