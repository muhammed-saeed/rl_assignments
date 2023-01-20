import argparse
import json
import os

parser = argparse.ArgumentParser(description='Reads environment json file.')
parser.add_argument('mode', type=str, help='train, val, or test')
parser.add_argument('--train-folder', type=str, help='path to the folder containing env json files')
parser.add_argument('--val-folder', type=str, help='path to the folder containing env json files')
parser.add_argument('--train-solution', type=str, help='path to the solution containing env json files')
parser.add_argument('--val-solution', type=str, help='path to the solution containing env json files')
parser.add_argument('--test-folder', type=str, help='path to the folder containing env json files')

args = parser.parse_args()

mode = args.mode
train_path = args.train_folder
train_target_path = args.train_solution
val_path = args.val_folder
val_target_path = args.val_solution


if (mode):
    if(train_path and train_target_path):
        json_files = [i for i in os.listdir(train_path) if i.endswith("json")]
        json_target_files = [i for i in os.listdir(train_target_path) if i.endswith("json")]
        # print(json_files)
        for file in json_files:
            if file in json_target_files: # if train example has a solution json included
                file_path = os.path.join(train_path, file)
                target_path = os.path.join(train_target_path, file)
                train_data = None
                train_seq = None
                
                with open(file_path) as f:
                    train_data = json.load(f)
                    
                with open(target_path) as f:
                    train_seq = json.load(f)
                    
                # print(train_data)
                rows = train_data["gridsz_num_rows"]
                cols = train_data["gridsz_num_cols"]
                agent_pos = (train_data["pregrid_agent_row"], train_data["pregrid_agent_col"])
                agent_dir = train_data["pregrid_agent_dir"]
                agent_final_pos = (train_data["postgrid_agent_row"], train_data["postgrid_agent_col"])
                agent_final_dir = train_data["postgrid_agent_dir"]
                walls = train_data["walls"]
                init_markers = train_data["pregrid_marker"]
                final_markers = train_data["postgrid_markers"]
                print(rows ,
                cols ,
                agent_pos ,
                agent_dir ,
                init_markers ,
                walls ,
                [agent_final_pos ,
                agent_final_dir ,
                final_markers ,
                ],
                ['m', 'l', 'r', 'f']
                )
                print()
                print(train_seq["sequence"])