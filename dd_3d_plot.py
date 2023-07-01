import matplotlib.pyplot as plt
import os
import re
import math

epochs = 4000

#directory = 'assets/MNIST/sub-set/epoch=%d-noise-5' % epochs
#plots_directory = os.path.join(directory, 'plots')
#input_file = os.path.join(directory, 'epoch=%d.txt' % epochs)

#if not os.path.isdir(plots_directory):
#    os.mkdir(plots_directory)

import csv

# Dictionary to be saved
person = {"name": "Jessa", "country": "USA", "telephone": 1178}
print('Person dictionary')
print(person)

# Open a csv file for writing
with open("person.csv", "w", newline="") as fp:
    # Create a writer object
    writer = csv.DictWriter(fp, fieldnames=person.keys())

    # Write the header row
    # writer.writeheader()

    # Write the data rows
    writer.writerow(person)
    print('Done writing dict to a csv file')