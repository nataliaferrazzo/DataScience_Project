import pandas as pd
import random
from helper_functions import train_test_split
from decision_tree_functions import decision_tree_algorithm, calculate_accuracy 
import pprint

df = pd.read_csv("creditcard.csv") #Carregando o dataset
df = df.rename(columns={df.columns[-1]: "label"}) #Renomeando última coluna
df = df.drop(["Time"], axis=1) #Excluindo colunas desnecessárias
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
amount = df['Amount'].values

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

random.seed(0)

train_df, test_df = train_test_split(df, 0.2)
tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=10)
accuracy = calculate_accuracy(test_df, tree)

pprint(tree, width=50)
accuracy