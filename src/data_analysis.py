import pandas as pd
import matplotlib.pyplot as plt

model_A = pd.read_csv("src/data/dataA.csv",names=["episode","avg_q_value"],header=None)
model_B = pd.read_csv("src/data/dataB.csv",names=["episode","avg_q_value"],header=None)
model_C = pd.read_csv("src/data/dataC.csv",names=["episode","avg_q_value"],header=None)
model_D = pd.read_csv("src/data/dataD.csv",names=["episode","avg_q_value"],header=None)
model_E = pd.read_csv("src/data/dataE.csv",names=["episode","avg_q_value"],header=None)

plt.plot(model_A["episode"], model_A["avg_q_value"], color='red',label="Model A")
plt.plot(model_B["episode"][:17], model_B["avg_q_value"][:17], color='yellow',label="Model B")
plt.plot(model_C["episode"][:17], model_C["avg_q_value"][:17], color='green',label="Model C")
plt.plot(model_D["episode"][:17], model_D["avg_q_value"][:17], color='blue',label="Model D")
plt.plot(model_E["episode"][:17], model_E["avg_q_value"][:17], color='black',label="Model E")
# Add title and labels
plt.title('Average Maximum Action Q-Values vs Number of episodes')
plt.xlabel('Number of Episodes')
plt.ylabel('Average Maximum Action Q-Values')
plt.legend()
# Save the plot to the 'plot' directory as a PNG file
plt.savefig('src/plots/model_all_training.png')