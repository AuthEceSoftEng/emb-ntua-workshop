import json
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from utilities import GesturesVisualizer as GV
from utilities import FeaturesExtractor as FE

def read_data():
    # Read data
    users = pd.read_csv("1_BrainRun/data/users.csv", sep=";")
    devices = pd.read_csv("1_BrainRun/data/devices.csv", sep=";")
    games = pd.read_csv("1_BrainRun/data/games.csv", sep=";")
    gestures = pd.read_csv("1_BrainRun/data/gestures.csv", sep=";")

    # Transform string representation of list into a python list
    gestures["data"] = gestures["data"].apply(lambda x: json.loads(x.replace("'", "\"")))

    return users, devices, games, gestures

def get_statistics(users, devices, games, gestures):
    # Get data statistics
    print("\n")
    print("Number of users: ", len(users.index))
    print("Number of devices: ", len(devices.index))
    print("Number of games: ", len(games.index))
    print("Number of gestures: ", len(gestures.index))

    # Print statistics about age, num_of_devices, num_of_games and num_of_gestures for each user
    print("\n", users[["age", "num_of_devices", "num_of_games", "num_of_gestures"]].describe())

    # Print statistics about devices os
    print("\n", devices[["os"]].describe())

    # Print statistics about game types
    print("\n", games[["game_type"]].describe())

    # Print statistics about gesture types
    print("\n", gestures[["type"]].describe())

def create_pie_charts(users, devices, games, gestures):

    # Percentage of playing time per game type
    types = games["game_type"].tolist()
    plt.subplot(221).pie(Counter(types).values(), labels=Counter(types).keys(), autopct='%1.1f%%', shadow=True)

    # Percentage of swipes and taps
    types = gestures["type"].tolist()
    plt.subplot(222).pie(Counter(types).values(), labels=Counter(types).keys(), autopct='%1.1f%%', shadow=True)

    # Percentage of male and female users
    types = users["gender"].tolist()
    plt.subplot(223).pie(Counter(types).values(), labels=Counter(types).keys(), autopct='%1.1f%%', shadow=True)

    # Percentage of male and female users
    types = devices["os"].tolist()
    plt.subplot(224).pie(Counter(types).values(), labels=Counter(types).keys(), autopct='%1.1f%%', shadow=True)
    
    plt.show()

def create_line_charts(users, gestures):

    # Plot users experience points
    # plt.subplot(211).plot(x=list(np.arange(1, len(users) + 1)), y=list(users["xp"]))
    plt.subplot(211).plot(list(np.arange(1, len(users) + 1)), list(users["xp"]), "-")
    plt.subplot(211).set_xlabel("Users")
    plt.subplot(211).set_ylabel("Experience points")
    plt.subplot(211).set_title("Experience points per user")

    # Plot number of data points sampled from each swipe
    swipes = gestures[(gestures["type"] == "swipe")]
    lengths = swipes["data"].apply(lambda x: len(x))
    
    plt.subplot(212).plot(list(np.arange(1, len(lengths) + 1)), list(lengths))
    plt.subplot(212).set_xlabel("Swipes")
    plt.subplot(212).set_ylabel("Number of data points")
    plt.subplot(212).set_title("Number of data points sampled from each swipe")
    
    plt.subplots_adjust(left = 0.15, top = 0.95, hspace = 0.55)
    plt.show()

def create_boxplot(devices):

    devices["os"] = devices.set_index("os").index
    devices.boxplot(column=["width", "height"], by="os", rot=45)
    plt.show()

def visualize_gestures(users, devices):

    uid = users[(users["_id"] == "5b5b2b94ed261d61ede3d085")].iloc[0]["_id"]
    devIds = devices[(devices["user_id"] == uid)].iloc[:]["device_id"]

    gests = pd.DataFrame()
    for devId in devIds.index:
        gests = gests.append(gestures[(gestures["device_id"] == devIds.loc[devId])])
    gests = gests.reset_index(drop=True)

    gestureVisualizer = GV.GesturesVisualizer(gests.loc[0:25], deviceWidth=411, deviceHeight=798)
    gestureVisualizer.plot_gestures()

def get_features(users, devices):

    # Get user info
    uid = users[(users["_id"] == "5b5b2b94ed261d61ede3d085")].iloc[0]["_id"]
    # Get devices info for the user
    devIds = devices[(devices["user_id"] == uid)].iloc[:]["device_id"]

    # Get gestures
    gests = pd.DataFrame()
    for devId in devIds.index:
        gests = gests.append(gestures[(gestures["device_id"] == devIds.loc[devId])])
    gests = gests.reset_index(drop=True)

    # Get calculated features for a certain swipe
    featuresExtractor = FE.FeaturesExtractor(gests)
    features_info = featuresExtractor.get_swipe_features(gests.loc[2])
    print(json.dumps(features_info, indent = 2))


# Read data
print("Reading data...")
users, devices, games, gestures = read_data()
print("Reading ended...")

# Get general statistics 
get_statistics(users, devices, games, gestures)

# Create pie charts 
create_pie_charts(users, devices, games, gestures)

# Create line charts 
create_line_charts(users, gestures)

# Create box plot
create_boxplot(devices)

# Create box plot
visualize_gestures(users, devices)

# Get Features
get_features(users, devices)
