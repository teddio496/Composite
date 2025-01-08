import joblib
import os
import pandas as pd
from app.data_constants import DATA_CONSTANT


def load_and_predict_comp(model_path: str, champion_ids: str):
    rf_model = joblib.load(model_path)

    input_data = pd.DataFrame([champion_ids], columns=[f"feature_{i}" for i in range(2, 12)])

    prediction = rf_model.predict(input_data)
    probabilities = rf_model.predict_proba(input_data)
    confidence = max(probabilities[0])  # The probability of the predicted class
    return prediction[0], confidence


def data_processor(champion_ids: list[int], data_dict=DATA_CONSTANT):
    """ALL INFO without champion names"""
    processed = []
    for i in range(0, 10):
        champ = (str(champion_ids[i]) + "-" + str(((i + 1) % 5)))
        if champ in data_dict:
            processed += [champion_ids[i]] + data_dict[champ][0: 8]
        else:
            processed += [0] * 9
    return processed


def load_and_predict_comp_processed(model_path: str, champion_ids: str):
    rf_model = joblib.load(model_path)

    input_data = pd.DataFrame([data_processor(champion_ids)], columns=["feature_" + str(i) for i in range(1, 91)])

    prediction = rf_model.predict(input_data)
    probabilities = rf_model.predict_proba(input_data)
    confidence = max(probabilities[0])  # The probability of the predicted class
    return prediction[0], confidence


def analyze_composition(champion_ids):
    current_dir = "app"
    model_path = os.path.join(current_dir, "random_forest_model_v1.pkl")
    model_path_processed = os.path.join(current_dir, "random_forest_model_preprocessed_v1.pkl")

    prediction_basic, confidence_basic = load_and_predict_comp(model_path, champion_ids)
    prediction_processed, confidence_processed = load_and_predict_comp_processed(model_path_processed, champion_ids)

    if abs(0.5 - confidence_basic) > abs(0.5 - confidence_processed):
        return int(prediction_basic)
    else:
        return int(prediction_processed)


if __name__ == "__main__":
    print(analyze_composition([i for i in range(1, 11)]))
    """

    # Testing
    list_of_data = []

    with open("data_test.csv", "r") as file:
        for line in file:
            row = line.strip().split(',')
            list_of_data.append([row[1]] + row[3:13])

    count = 0
    correct_count = 0

    for i in list_of_data:
        print(count)
        count += 1

        prediction_basic, confidence_basic = load_and_predict_comp(model_path, i[1:])
        prediction_procesed, confidence_processed = load_and_predict_comp_processed(model_path_processed, i[1:])

        if abs(0.5 - confidence) > abs(0.5 - confidence_processed):
            correct_count += 1 if int(i[0]) == int(prediction_basic) else 0
        else:
            correct_count += 1 if int(i[0]) == int(prediction_procesed) else 0


    print(count)
    print(correct_count)
    print(second_count)
    print(both_correct)
    print(both_correct / count)
    print(correct_count / count)

    """
