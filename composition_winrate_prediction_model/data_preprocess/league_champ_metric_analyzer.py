from constants import CHAMP_ID_TO_NAME, STATS, CHALLENGE_STATS
# from tabulate import tabulate
import json

import csv
# import pandas as pd


def champion_id_to_champion_name(champId: int) -> str:
    return CHAMP_ID_TO_NAME[champId]


def champion_name_to_champion_id(champName: str) -> int:
    for key, value in CHAMP_ID_TO_NAME.items():
        if value == champName:
            return key


def create_champ_metric(data_file: str, num_stats: int, readable: bool = False):
    with open(data_file, mode='r') as file:
        csv_reader = csv.reader(file)
        data_dict = {}
        lane_map = ["TOP", "JG", "MID", "ADC", "SUP"]
        for row in csv_reader:
            row_data = row[3:]
            for i in range(0, 10):
                id_value = row_data[i]
                info_columns = row_data[(i * num_stats) + 10:((i+1) * num_stats) + 10 ] + [1]
                info_columns = list(map(float, info_columns))
        
                key = champion_id_to_champion_name(int(id_value)) + "-" + lane_map[i % 5] if readable \
                    else id_value + "-" + str(i % 5)
                
                current_data = data_dict.get(key, [])
                if current_data:
                    for j in range(len(info_columns)):
                        info_columns[j] += current_data[j] 

                data_dict[key] = info_columns

    for key in data_dict:
        total_count = data_dict[key][-1]
        data_dict[key] = [(value // total_count) for value in data_dict[key][:-1]] + [total_count]

    return data_dict


def apply_champ_metric(data_file: str, write_file: str, processor):
    # Create the champion metric dictionary (this part assumes you have create_champ_metric defined)
    data_dict = create_champ_metric('data.csv', len(STATS) + len(CHALLENGE_STATS), readable=False)

    with open(data_file, mode='r') as file:
        csv_reader = csv.reader(file)

        with open(write_file, mode="w", newline="") as write:
            csv_writer = csv.writer(write)
            
            for row in csv_reader:
                processed_metric = processor(row, data_dict)
                new_row = processed_metric

                csv_writer.writerow(new_row)


def data_processor_1(row: list, data_dict: dict):
    """ALL INFO with champion names"""
    processed = [row[1]]
    for i in range(3, 13):
        champ = (row[i] + "-" + str(((i-2) % 5)))
        if champ in data_dict:
            processed += [row[i]] + data_dict[champ][0: 8]
        else:
            print(champ)
            processed += [0] * 9
    return processed


def data_processor_2(row: list, data_dict: dict):
    """ALL INFO without champion names"""
    processed = [row[1]]
    for i in range(3, 13):
        champ = (row[i] + "-" + str(((i-2) % 5)))
        if champ in data_dict:
            processed += data_dict[champ][0: 8]
        else:
            processed += [0* 10]
    return processed


def data_processor_3(row: list, data_dict: dict):
    """Combined info with champion names"""
    processed = [row[1]]
    team_metric_1 = [0] * 8
    team_metric_2 = [0] * 8
    
    for i in range(3, 8):
        champ = (row[i] + "-" + str(((i-3) % 5)))
        processed += [row[i]]    
        if champ in data_dict:
            for j in range(len(data_dict[champ][:-1])):
                team_metric_1[j] += data_dict[champ][j]

    for i in range(8, 13):
        champ = (row[i] + "-" + str(((i-3) % 5)))
        processed += [row[i]]    
        for k in range(len(data_dict[champ][:-1])):
            team_metric_2[k] += data_dict[champ][k]
            
    return processed + team_metric_1 + team_metric_2


def data_processor_4 (row: list, data_dict: dict):
    """Combined info without champion names"""
    processed = [row[1]]
    team_metric_1 = [0] * 8
    team_metric_2 = [0] * 8
    
    for i in range(3, 8):
        champ = (row[i] + "-" + str(((i-3) % 5)))
        if champ in data_dict:
            for j in range(len(data_dict[champ][:-1])):
                team_metric_1[j] += data_dict[champ][j]

    for i in range(8, 13):
        champ = (row[i] + "-" + str(((i-3) % 5)))
        for k in range(len(data_dict[champ][:-1])):
            team_metric_1[k] -= data_dict[champ][k]
            
    return processed + team_metric_1


if __name__ == "__main__":
    apply_champ_metric("data.csv", "preprocessed.csv", data_processor_1)
    #data_dict = create_champ_metric('data.csv', len(STATS) + len(CHALLENGE_STATS), readable=False)
    #constant_content = f"DATA_CONSTANT = {json.dumps(data_dict, indent=4)}"

    #with open("data_constants.py", "w") as f:
    #    f.write(constant_content)
    """
    

    #data display
    data_frame = pd.DataFrame.from_dict(
        data_dict, 
        orient = 'index', 
        columns = STATS + CHALLENGE_STATS +["Count"]
    )

    data_frame.reset_index(inplace=True)
    data_frame.rename(columns={"index": "Champion_ID"}, inplace=True)

    print(tabulate(data_frame, headers="keys", tablefmt="grid"))
    with open("champ_stats.txt", "w") as file:
        file.write(tabulate(data_frame, headers="keys", tablefmt="grid"))
    """