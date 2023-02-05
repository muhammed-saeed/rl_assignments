import os
import json


# val_target_path = args.val_solution
def read_env_sol_json (mode,train_path,train_target_path):
    ENV = []
    SEQ = []
    FILES= []
    if(train_path and train_target_path):
        json_files = [i for i in os.listdir(train_path) if i.endswith("json")]
        json_target_files = [i for i in os.listdir(train_target_path) if i.endswith("json")]
        print(json_files)
        print(json_target_files)
        for counter, file in enumerate(json_files):
            target_file = file.split("task")[0]+"seq.json"
            if target_file in json_target_files: 
            # if file in json_target_files: # if train example has a solution json included
                file_path = os.path.join(train_path, file)
                # target_path = os.path.join(train_target_path, json_target_files[counter])
                target_path = os.path.join(train_target_path, target_file)
                train_data = None
                train_seq = None
                
                with open(file_path) as f:
                    train_data = json.load(f)
                with open(target_path) as f:
                    train_seq = json.load(f)
                # print(train_data)
                # print(train_data)
                # print(train_data)
                rows = train_data["gridsz_num_rows"]
                cols = train_data["gridsz_num_cols"]
                agent_pos = (train_data["pregrid_agent_row"], train_data["pregrid_agent_col"])
                agent_dir = train_data["pregrid_agent_dir"]
                agent_final_pos = (train_data["postgrid_agent_row"], train_data["postgrid_agent_col"])
                agent_final_dir = train_data["postgrid_agent_dir"]
                walls = train_data["walls"]
                init_markers = train_data["pregrid_markers"]
                final_markers = train_data["postgrid_markers"]
                # print(type(walls[0]))
                # print(walls)
                walls_tuple = [tuple(sublist) for sublist in walls]
                init_markers_tuple = [tuple(sublist) for sublist in init_markers]
                final_markers_tuple = [tuple(sublist) for sublist in final_markers]
                enviromentDetails =  (rows ,
                cols ,
                agent_pos ,
                agent_dir ,
                init_markers_tuple ,
                walls_tuple ,
                [agent_final_pos ,
                agent_final_dir ,
                final_markers_tuple ,
                ],
                ['m', 'l', 'r', 'f','pick','put']
                )
                # print()
                # print(train_seq["sequence"])
                # print(train_seq["sequence"])
                bestActionSeq = train_seq["sequence"]
                # return (enviromentDetails, bestActionSeq)
                ENV.append(enviromentDetails)
                SEQ.append(bestActionSeq)
                FILES.append(file.split("task")[0]+'task')
    
    return ENV, SEQ,FILES
                
