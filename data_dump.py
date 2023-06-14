import pymongo
import pandas as pd
import json

client = pymongo.MongoClient("mongodb+srv://sambhavm22:sambhav@cluster0.oekbqjn.mongodb.net/?retryWrites=true&w=majority")

DATABASE_NAME = 'INSURANCE'
COLLECTION_NAME = 'INSURANCE_PROJECT'

if __name__ == "__main__":
    df = pd.read_csv('/Users/aakanksha/My_Codes/Project-EWB/insurance.csv')
    print(f"Rows and Columns:{df.shape}")

    df.reset_index(drop=True, inplace=True)
    
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    