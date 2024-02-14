import bz2
import json
import pickle

with bz2.BZ2File('drop_tables.pkl', 'rb') as f:
    dt = pickle.load(f)


# open json file 
with open('json_data/train_tables.jsonl', 'r') as f:
    # read the first table and check if id is in the list of ids to remove
    # do for the first 10 tables
    while True:
        line = f.readline()
        if not line:
            print('------------------READ FINISCHED------------------')
            break
        table = json.loads(line)
        if table['_id'] in dt:
            print(f"Removing table with id {table['_id']}")
            continue
        # write the table to a new file
        with open('json_data/train_tables_filtered.jsonl', 'a') as f2:
            f2.write(json.dumps(table) + '\n')
