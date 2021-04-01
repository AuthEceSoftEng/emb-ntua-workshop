import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt

class GesturesVisualizer():

    def __init__(self, gestures, deviceWidth=360, deviceHeight=640):
        self.gestures = gestures
        self.width = deviceWidth
        self.height = deviceHeight

    def plot_gestures(self):
        fig = plt.figure(figsize=(3.75, 2.5 * (self.height / self.width)))
        ax = fig.add_axes([0.15, 0.05, 0.55, 0.85])
        labels = OrderedDict()
        for i, _ind in enumerate(self.gestures.index):
            labels["gesture_" + str(i)] = np.random.rand(1, 3)
            x_data = []
            y_data = []
            if(len(self.gestures.iloc[i]["data"]) == 0):
                continue
            x_data.append(self.gestures.iloc[i]["data"][0]["x0"])
            y_data.append(self.gestures.iloc[i]["data"][0]["y0"])
            if(self.gestures.iloc[i]["type"] == "swipe"):
                for d in self.gestures.iloc[i]["data"]:
                    x_data.append(d["moveX"])
                    y_data.append(d["moveY"])
            keys = list(labels.keys())
            if(self.gestures.iloc[i]["type"] == "tap"):
                plt.scatter(x_data, y_data, label=keys[i], color = labels[keys[i]][0])
            else:
                plt.plot(x_data, y_data, label=keys[i], color = labels[keys[i]][0])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xlabel('X - Dimension')
        plt.ylabel('Y - Dimension')
        plt.gca().invert_yaxis()
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 0.5), loc="center left")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        plt.show()