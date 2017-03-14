import pandas as pd
from matplotlib import *
from matplotlib.pyplot import *

train_log = pd.read_csv("../logs/logteeth.log.train")
test_log = pd.read_csv("../logs/logteeth.log.test")
_, ax1 = subplots(figsize=(15, 10))
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
savefig("../train_test_image.png") 