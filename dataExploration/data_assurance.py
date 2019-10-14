import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
Q = question
A = answer
O = observation
"""

dir_ = '/Users/lukeprice/Codes/BrainCorp/'
dat = pd.read_csv(dir_ + 'device_failure_data_scientist.csv')
dat = dat.sort_values(by = ['device', 'date'])
# # O: the date is not YYYY-MM-DD. It appears to be simply be add int(1) for each new day.

# Q: Any missing values?
print(all(dat.isnull()))
input()
# A: No. 

# sets of devices and device counts 
devices = set(dat.loc[:,'device'])
device_counts = Counter(dat.loc[:,'device'])
# # number of devices in use by day 
plt.hist(device_counts.values(), len(set(device_counts.values())), facecolor = 'blue', alpha = 0.5)
plt.show()

#----------------------------------------------------------------
# Q: can a device fail more than once in a single day?
max_daily_failures = max(dat.loc[:,'failure'])
print('\nMax number of individual failures in single day: {}'.format(max_daily_failures))
input()
# A: no


# Q: Can a device fail more than once?
num_failures = {}
with tqdm(total = len(device_counts)) as pbar: 
  for device in device_counts.keys():
    device_table = dat.loc[dat['device'] == device]
    num_failures[device] = np.sum(device_table.loc[:,'failure'])
    pbar.update(1) 
print('\nSet of number of times a machine failed: {}'.format(set(num_failures.values())))
input()
# A: No


# Q: if a device fails, can it be put back in use (i.e. when it fails, is it permanently out of rotation)?
last_day_failures = 0
re_used_IDs = []
with tqdm(total = len(device_counts)) as pbar: 
  for device in device_counts.keys():
    outcome = list(dat.loc[dat['device'] == device].loc[:,'failure'])
    if 1 in outcome:
      if outcome[-1] == 1:
        last_day_failures += 1
      else: 
        re_used_IDs.append(device)
    pbar.update(1) 

print('\nNumber of machines NOT re-used after failures: {}'.format(last_day_failures))
print('Total number of failed machines: {}'.format(np.sum(dat.loc[:,'failure'])))
print('Re-used machine IDs: {}'.format(re_used_IDs))
input()
# A: Yes. 5 difference devices fail and are put back into use on future dates 


# Q: what does the failures-by-date plot look like?
fail_by_date = dat[['date', 'failure']].groupby(['date']).sum()
max_failures = max(fail_by_date.loc[:,'failure'])
fail_by_date.unstack().plot()
plt.show()
# O: does not appear to be a trend, drift, of change in variance; spike at 
print(fail_by_date.loc[fail_by_date['failure'] == max_failures])
input()
# O: spike at date 15019 with 8 failures 


# Q: Are all machine's first day on day_1 (15001)?
dates = dat.loc[:,'date']
day_1 = min(set(dates))
first_days = {}
with tqdm(total = len(devices)) as pbar: 
  for device in devices:
    device_dat = dat.loc[dat['device'] == device]
    first_days[device] = min(device_dat.loc[:,'date'])
    pbar.update(1) 
print(set(first_days.values()))
print(Counter(first_days.values()))
input()
# A: No. 4 day 1 on 15125, 1 day 1 on 15027. 


# Q: what machines do not start on day 1 (15001)?
late_start_machines = []
for machine, day in first_days.items():
  if day != day_1:
    late_start_machines.append(machine)
print('Machines not started on day {} (day 1): {}'.format(day_1, late_start_machines))
# O: 5 machines not started on day 1 are disjoint from the 5 machines that are re-used after failure. 
# O: possibly remove these 5+5=10 machines from analysis
input()

# remove machines that are not started on on day 1 or if they are re-used
print(re_used_IDs)
print(late_start_machines)
remove_machines = re_used_IDs + late_start_machines
q1 = dat[~dat['device'].isin(remove_machines)]






