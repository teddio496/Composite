import requests
import time
from collections import deque
import csv
from constants import SHORT_TERM_LIMIT, LONG_TERM_LIMIT, COUNTRY_TO_REGION, STATS, CHALLENGE_STATS, RIOT_API_KEY

REQUEST_COUNT_SHORT = 0
REQUEST_COUNT_LONG = 0
LAST_RESET_SHORT = time.time()
LAST_RESET_LONG = time.time()


def riot_make_api_request(url: str):
    """
    Make a GET request to the Riot Games API, handling rate limits and retries.

    Args:
        url (str): The URL to which the GET request will be sent.

    Returns:
        dict or None: The response JSON if successful, None if an error occurred.
    """
    global REQUEST_COUNT_SHORT, REQUEST_COUNT_LONG, LAST_RESET_SHORT, LAST_RESET_LONG

    current_time = time.time()

    if current_time - LAST_RESET_SHORT >= 1:
        REQUEST_COUNT_SHORT = 0
        LAST_RESET_SHORT = current_time

    if current_time - LAST_RESET_LONG >= 120:
        REQUEST_COUNT_LONG = 0
        LAST_RESET_LONG = current_time

    if REQUEST_COUNT_SHORT >= SHORT_TERM_LIMIT:
        sleep_time = 1 - (current_time - LAST_RESET_SHORT)
        print(f"Short-term limit reached. Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(max(0, sleep_time))

    if REQUEST_COUNT_LONG >= LONG_TERM_LIMIT:
        sleep_time = 120 - (current_time - LAST_RESET_LONG)
        print(f"Long-term limit reached. Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(max(0, sleep_time))

    response = requests.get(url)
    REQUEST_COUNT_SHORT += 1
    REQUEST_COUNT_LONG += 1

    if response.status_code == 429:
        print("Rate Limit Exceeded, retrying after sleep...")
        time.sleep(1)
        return riot_make_api_request(url)
    elif response.status_code in {400, 401, 403, 404, 405, 415, 500, 502, 503, 504}:
        print(f"HTTP Error: {response.status_code}")
        return None
    else:
        return response.json()


def matchId_to_match(match_id: str, api_key: str, country: str):
    """
    Retrieve match data for a specific match ID from the Riot Games API.

    Args:
        match_id (str): The unique identifier for the match.
        api_key (str): The Riot Games API key.
        country (str): The country/region code.

    Returns:
        dict or None: The match data if successful, None if an error occurred.
    """
    region = COUNTRY_TO_REGION.get(country)
    if not region:
        raise ValueError(f"Invalid country code: {country}")
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}"
    return riot_make_api_request(url)


def player_to_match_ids(puuid: str, api_key: str, amount: int, start: int, country: str):
    """
    Retrieve a list of match IDs for a player identified by their PUUID.

    Args:
        puuid (str): The player's PUUID (Player Unique ID).
        api_key (str): The Riot Games API key.
        amount (int): The number of match IDs to retrieve.
        start (int): The starting point for the match IDs.
        country (str): The country/region code.

    Returns:
        list or None: A list of match IDs if successful, None if an error occurred.
    """
    region = COUNTRY_TO_REGION.get(country)
    if not region:
        raise ValueError(f"Invalid country code: {country}")
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&start={start}&count={amount}&api_key={api_key}"
    return riot_make_api_request(url)


def bfs_get_match_ids(amount: int, start_match_id: str, already: dict[str, int], api_key: str, country: str, output_file: str):
    """
    Perform a breadth-first search (BFS) to gather match IDs starting from a given match ID.

    Args:
        amount (int): The number of match IDs to retrieve.
        start_match_id (str): The starting match ID for the BFS.
        already (dict): A dictionary to track match IDs that have already been processed.
        api_key (str): The Riot Games API key.
        country (str): The country/region code.
        output_file (str): The file path to output the match IDs.

    Returns:
        list: A list of match IDs gathered from the BFS.
    """
    region = COUNTRY_TO_REGION.get(country)

    if not region:
        raise ValueError(f"Invalid country code: {country}")

    with open(output_file, "a") as f:
        counter = 0
        ret_matches = []
        queue = deque([start_match_id])

        while queue and counter < amount:
            current_match_id = queue.popleft()
            match_data = matchId_to_match(current_match_id, api_key, country)
            if not match_data:
                continue

            participants = match_data["metadata"]["participants"]

            for player_puuid in participants:
                summoner_url = f"https://{country}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{player_puuid}?api_key={api_key}"
                summoner_data = riot_make_api_request(summoner_url)
                if not summoner_data:
                    continue

                summoner_id = summoner_data["id"]
                rank_url = f"https://{country}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}?api_key={api_key}"
                rank_data = riot_make_api_request(rank_url)
                if not rank_data or not rank_data[0].get('tier'):
                    continue

                rank = rank_data[0]['tier']
                if rank not in {"CHALLENGER", "GRANDMASTER"}:
                    continue

                recent_matches = player_to_match_ids(player_puuid, api_key, 50, 0, country)
                if not recent_matches:
                    continue

                for match_id in recent_matches:
                    if match_id not in already:
                        already[match_id] = 1
                        queue.append(match_id)
                        ret_matches.append(match_id)
                        f.write(match_id + "\n")
                        print(f"{counter}: {match_id}")
                        counter += 1

    return ret_matches


def get_recent_match(gameName: str, tagline: str, api_key: str, country: str) -> str:
    """
    Retrieve the most recent match ID for a player based on their Riot account ID.

    Args:
        gameName (str): The player's in-game name.
        tagline (str): The player's in-game tagline.
        api_key (str): The Riot Games API key.
        country (str): The country/region code.

    Returns:
        str: The match ID of the most recent match.
    """
    region = COUNTRY_TO_REGION.get(country)
    if not region:
        raise ValueError(f"Invalid country code: {country}")

    puuid_url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{gameName}/{tagline}?api_key={api_key}"
    puuid = riot_make_api_request(puuid_url)["puuid"]

    match_url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count=1&api_key={api_key}"
    match = riot_make_api_request(match_url)[0]

    return match


def fetch_match_data(match_id: str, api_key: str, region: str) -> dict:
    """
    Retrieve match data for a specific match ID.

    Args:
        match_id (str): The match ID.
        api_key (str): The Riot Games API key.
        region (str): The country/region code.

    Returns:
        dict or None: The match data if successful, None if an error occurred.
    """
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}"
    return riot_make_api_request(url)


def process_match_data(match_info: dict) -> list:
    """
    Process match data into a structured format for CSV.

    Args:
        match_info (dict): The raw match data.

    Returns:
        list: A list containing processed match data ready for CSV writing.
    """
    if not match_info or "info" not in match_info:
        return ["error"]

    participants = match_info["info"].get("participants", [])
    if not participants:
        return ["error"]

    processed_participants = []
    processed_stats = []
    for participant in participants:
        processed_participants.append(participant["championId"])

        for stat in STATS:
            processed_stats.append(participant[stat])
        for c_stat in CHALLENGE_STATS:
            processed_stats.append(participant["challenges"][c_stat])

        print(participant["championName"], processed_stats[-8:])
        
    winning_team = 100 if participants[0].get("win", False) else 200

    return [match_info["metadata"]["matchId"], winning_team] + [match_info["info"].get("gameDuration", 0)] + processed_participants + processed_stats


def write_to_csv_row(row: list, file_name: str) -> None:
    """
    Write a row of data to a CSV file.

    Args:
        row (list): The data to write as a row.
        file_name (str): The CSV file to write to.
    """
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def convert_match_ids_to_csv(match_ids: list[str], api_key: str, file_name: str, country: str) -> None:
    """
    Convert a list of match IDs to CSV format by fetching match data.

    Args:
        match_ids (list): The list of match IDs.
        api_key (str): The Riot Games API key.
        file_name (str): The file to save the match data in CSV format.
        country (str): The country/region code.
    """
    num_error_rows = 0
    for count, match_id in enumerate(match_ids, start=1):
        match_info = fetch_match_data(match_id, api_key, COUNTRY_TO_REGION[country])
        if not match_info:
            print(f"Skipping match {match_id} due to errors.")
            continue

        processed_row = process_match_data(match_info)
        if processed_row[0] == "error":
            num_error_rows += 1
            print(f"Error Occured on row {count}, and match_id {match_id}")

        write_to_csv_row(processed_row, file_name)

        print(f"Processed match {count}/{len(match_ids)}: {match_id}")


def convert_txt_to_list(file_name):
    """
    Convert a text file of match IDs to a list.

    Args:
        file_name (str): The file containing match IDs.

    Returns:
        list: A list of match IDs.
    """
    with open(file_name, "r") as this:
        data = this.readlines()
    new = []
    for i in data:
        new.append(i[:-1])
    return new


def convert_txt_to_dict(file_name):
    """
    Convert a text file of match IDs to a dictionary for quick lookup.

    Args:
        file_name (str): The file containing match IDs.

    Returns:
        dict: A dictionary of match IDs.
    """
    already = {}
    for id in convert_txt_to_list(file_name):
        already[id] = 1
    return already


def get_name_and_id(match_ids: list[str], api_key: str, country: str) -> None:
    """
    Retrieve and print champion names and IDs for each participant in a list of match IDs.

    Args:
        match_ids (list): A list of match IDs.
        api_key (str): The Riot Games API key.
        country (str): The country/region code.
    """
    id_to_name = {}
    for match_id in match_ids:
        match_info = fetch_match_data(match_id, api_key, COUNTRY_TO_REGION[country])
        participants = match_info["info"].get("participants", [])
        for participant in participants:
            if participant["championId"] not in id_to_name:
                id_to_name[participant["championId"]] = participant["championName"]
                print(participant["championId"], participant["championName"])
        print(id_to_name)
        print(len(id_to_name))


if __name__ == "__main__":
    API_KEY = RIOT_API_KEY
    COUNTRY = "NA1"
    MATCH_IDS_FILE = "output.txt"
    DATA_FILE = "data.csv"

    # already = convert_txt_to_dict("match_ids.txt")
    # recent_match = get_recent_match("ASTROBOY99", "NA1", API_KEY, COUNTRY)
    # bfs_get_match_ids(3000, recent_match, already, API_KEY, COUNTRY, MATCH_IDS_FILE)

    match_ids = convert_txt_to_list("output.txt")
    convert_match_ids_to_csv(match_ids, API_KEY, "data_test.csv", COUNTRY)

    #get_name_and_id(match_ids, API_KEY, COUNTRY)

