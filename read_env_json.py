import argparse
import json

parser = argparse.ArgumentParser(description='Reads environment json file.')
parser.add_argument('filename', type=str, help='path to env json file')

args = parser.parse_args()

path = args.filename
obj = None
with open(path) as f:
    obj = json.load(f)
    
if(obj):
    rows = obj["gridsz_num_rows"]
    cols = obj["gridsz_num_cols"]
    agent_pos = (obj["pregrid_agent_row"], obj["pregrid_agent_col"])
    agent_dir = obj["pregrid_agent_dir"]
    agent_final_pos = (obj["postgrid_agent_row"], obj["postgrid_agent_col"])
    agent_final_dir = obj["postgrid_agent_dir"]
    walls = obj["walls"]
    init_markers = obj["pregrid_marker"]
    final_markers = obj["postgrid_markers"]
    # print(rows ,
    # cols ,
    # agent_pos ,
    # agent_dir ,
    # init_markers ,
    # walls ,
    # [agent_final_pos ,
    # agent_final_dir ,
    # final_markers ,
    # ],
    # ['m', 'l', 'r', 'f']
    # )