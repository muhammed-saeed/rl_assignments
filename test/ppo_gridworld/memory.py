import json
import os
from env import GridWorld

convert_moves_dict = {
    'move': "m",
    'turnLeft': "l",
    'turnRight': "r",
    'pickMarker': "pick",
    'finish': "f",
    'putMarker': "put"
}

actions = ['m', 'l', 'r', 'f', "pick", "put"]

def get_memory(mode="train", train_path="/home/CE/musaeed/rl_assignments/train", train_target_path="/home/CE/musaeed/rl_assignments/trainSolution"): 
    memory = []   
    if (mode):
        if (train_path and train_target_path):
            json_files = [i for i in os.listdir(train_path) if i.endswith("json")]
            # print(json_files)
            json_target_files = [i for i in os.listdir(
                train_target_path) if i.endswith("json")]
            # print(json_files)
            for file in json_files:
                target_file = file.split("task")[0]+"seq.json"
                if target_file in json_target_files:  # if train example has a solution json included
                    # print(file, target_file)
                    file_path = os.path.join(train_path, file)
                    target_path = os.path.join(train_target_path, target_file)
                    train_data = None
                    train_seq = None

                    with open(file_path) as f:
                        train_data = json.load(f)

                    with open(target_path) as f:
                        train_seq = json.load(f)

                    # print(train_data)
                    rows = train_data["gridsz_num_rows"]
                    cols = train_data["gridsz_num_cols"]
                    agent_pos = (train_data["pregrid_agent_row"],
                                train_data["pregrid_agent_col"])
                    agent_dir = train_data["pregrid_agent_dir"]
                    agent_final_pos = (
                        train_data["postgrid_agent_row"], train_data["postgrid_agent_col"])
                    agent_final_dir = train_data["postgrid_agent_dir"]
                    walls = [(i, j) for [i, j] in train_data["walls"]]
                    init_markers = [(i, j)
                                    for [i, j] in train_data["pregrid_markers"]]
                    final_markers = [(i, j)
                                    for [i, j] in train_data["postgrid_markers"]]
                    # print(rows,
                    #     cols,
                    #     agent_pos,
                    #     agent_dir,
                    #     init_markers,
                    #     walls,
                    #     [agent_final_pos,
                    #     agent_final_dir,
                    #     final_markers,
                    #     ],
                    #     ['m', 'l', 'r', 'f', "pick", "put"]
                    #     )
                    env = GridWorld(rows,
                                    cols,
                                    agent_pos,
                                    agent_dir,
                                    init_markers,
                                    walls,
                                    [agent_final_pos,
                                    agent_final_dir,
                                    final_markers,
                                    ],
                                    ['m', 'l', 'r', 'f', "pick", "put"]
                                    )
                    # print()
                    seq = [convert_moves_dict[i] for i in train_seq["sequence"]]
                    # print(train_seq["sequence"])
                    # print(seq)
                    state = env.reset()
                    for i in seq:
                        next_state, reward, done, _ = env.step(i)
                        dead_win = reward >-1
                        memory.append((state, actions.index(i), reward, next_state, 1., done, dead_win))
                        state = next_state
    return memory
