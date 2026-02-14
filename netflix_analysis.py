import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# Show first 5 rows
print(df.head())

# Check dataset info
print(df.info())

# Check missing values
print(df.isnull().sum())
