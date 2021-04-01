import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

class FeaturesExtractor():
    def __init__(self, gestures, fake_swipes_limit=30):        
        gestures = gestures[(gestures["t_stop"] != -1) & (gestures["t_start"] != -1)]
        gestures["duration"] = gestures["t_stop"] - gestures["t_start"]
        gestures = gestures[(gestures["duration"] > 0)]

        self.taps = gestures[(gestures["type"] == "tap")]
        self.swipes = gestures[(gestures["type"] == "swipe") & (gestures["duration"] >= fake_swipes_limit)]
        self.fake_swipes = gestures[(gestures["type"] == "swipe") & (gestures["duration"] < fake_swipes_limit)]
        
        print("\n==== Gestures Stats ====")
        print("Taps: ", len(self.taps.index))
        print("Swipes: ", len(self.swipes.index))
        print("Fake swipes: ", len(self.fake_swipes.index), "\n")

    def get_tap_features(self, tap):
        info = {}
        info["type"] = "tap"
        info["horizontal_position"] = tap["data"][0]["x0"]
        info["vertical_position"] = tap["data"][0]["y0"]
        return info

    def get_swipe_features(self, swipe):
        info = {}
        info["type"] = "swipe"

        times = []
        num_data = len(swipe["data"])+1
        if(num_data==2):
            times.append(swipe["t_start"])
            times.append(swipe["t_stop"])
        elif(num_data>2):
            step = (swipe["t_stop"]-swipe["t_start"])/(num_data-1)
            times.append(swipe["t_start"])
            prev = swipe["t_start"]
            for i in range(0,num_data-2):
                times.append(prev+step)
                prev += step
            times.append(swipe["t_stop"])
        for i in range(0,len(times)):
            times[i] -= swipe["t_start"]
        
        x_positions = []
        y_positions = []
        # Get horizontal and vertical starting points
        x_positions.append(swipe["data"][0]["x0"])
        y_positions.append(swipe["data"][0]["y0"])

        for d in swipe["data"]:
            x_positions.append(d["moveX"])
            y_positions.append(d["moveY"])

        horizontal_length = x_positions[-1] - x_positions[0]
        vertical_length = y_positions[-1] - y_positions[0]
        info["horizontal_trace_length"] = np.abs(horizontal_length)
        info["vertical_trace_length"] = np.abs(vertical_length)
        if(np.abs(horizontal_length)>np.abs(vertical_length)):
            if(horizontal_length>0):
                info["direction"] = "right"
            else:
                info["direction"] = "left"
        else:
            if(vertical_length>0):
                info["direction"] = "up"
            else:
                info["direction"] = "down"

        # Get statistics of trace
        info["trace_stats"] = self.perform_linear_regression(x_positions, y_positions)

        info["swipe_horizontal_acceleration"] = (swipe["data"][-1]["vx"] - swipe["data"][0]["vx"])/((swipe["t_stop"] - swipe["t_start"])*0.001)
        info["swipe_vertical_acceleration"] = (swipe["data"][-1]["vy"] - swipe["data"][0]["vy"])/((swipe["t_stop"] - swipe["t_start"])*0.001)

        mean_x = 0
        mean_y = 0
        for x in x_positions:
            mean_x += x
        for y in y_positions:
            mean_y += y
        mean_x /= len(x_positions)
        mean_y /= len(y_positions)
        info["mean_x"] = mean_x
        info["mean_y"] = mean_y
        
        return info

    def get_fake_swipe_features(self, fake_swipe):
        info = {}
        info["type"] = "fake_swipe"

        info["fs_horizontal_position"] = fake_swipe["data"][0]["x0"]
        info["fs_vertical_position"] = fake_swipe["data"][0]["y0"]

        return info

    def calculate_features(self):
        features = []
        for ind in self.taps.index:
            features.append(self.get_tap_features(self.taps.loc[ind]))
        for ind in self.swipes.index:
            features.append(self.get_swipe_features(self.swipes.loc[ind]))
        for ind in self.fake_swipes.index:
            features.append(self.get_fake_swipe_features(self.fake_swipes.loc[ind]))

        return features

    def perform_linear_regression(self, x_pos, y_pos):
        
        x_train = np.array(x_pos).reshape(-1, 1)
        y_train = np.array(y_pos).reshape(-1, 1)
        
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(x_train, y_train)
        # Predict based on the constructed model
        pred = regr.predict(x_train)
        
        info = {}
        info["slope"] = regr.coef_[0][0]
        info["mean_squared_error"] = mean_squared_error(y_train, pred)
        info["mean_abs_error"] = mean_absolute_error(y_train, pred)
        info["median_abs_error"] = median_absolute_error(y_train, pred)
        info["coef_determination"] = r2_score(y_train, pred)
        
        return info
