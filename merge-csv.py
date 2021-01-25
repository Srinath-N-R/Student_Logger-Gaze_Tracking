import pandas as pd

all_filenames = ['data{}.txt'.format(i) for i in range(1,11)]

combined_csv = pd.concat([pd.read_csv(f, sep = ' ', header=None) for f in all_filenames ])
combined_csv.to_csv( "data.txt", index=False, encoding='utf-8-sig', sep = ' ', header=None)