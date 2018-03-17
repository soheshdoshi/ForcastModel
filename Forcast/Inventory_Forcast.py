import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize


columns = defaultdict(list) # each value in each column is appended to a list

with open('Inventory1.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list

                    # based on column name k

Mon=columns['Mon']  #read which colum name is monday
Tue=columns['Tue']
Wed=columns['Wed']
Thu=columns['Thu']
Fri=columns['Fri']
Sat=columns['Sat']
Sun=columns['Sun']
results_Mon = map(int,Mon)  #here map convet string list to int
np_result_Mon=np.array(results_Mon) #here np array can convert list to array


def mean(numbers): #function for mean when data is level
    return float(sum(numbers)) / max(len(numbers), 1)

def average(series): #function for average when data is level
    return float(sum(series))/len(series)

def moving_average(series, n): #function for moving_avg when data is level
    return average(series[-n:])

def weighted_average(series, weights): #function for weighted_avger when data is level
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result

def exponential_smoothing(series, alpha):  #function for exponetial smoothing foe level data
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def double_exponential_smoothing(series, alpha, beta):  #function for level and trade data
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def initial_trend(series, slen):    #find for initial trend
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_seasonal_components(series, slen): #function for seasonal component
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

 #function for holt'winter model when data is level+trend+seasonal
def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result


def sse(x): #function for sum of squared errors of prediction
    pred = triple_exponential_smoothing(np_result_Mon, 12, x[0], x[1], x[2], 24)
    return sum((pred[:np_result_Mon.shape[0]] - np_result_Mon)**2)



#print (results_Mon)

Error_fun=minimize(sse,[0.5, 0.5, 0.5], method='SLSQP', bounds=[(0,1), (0,1), (0,1)])
print (Error_fun)
# holtwinters(results_Mon, 0.2, 0.1, 0.05, 4)

avg_Mon=np.append(results_Mon,average(results_Mon))
#print (avg_Mon)
mean_Mon=np.append(results_Mon,mean(results_Mon))
#print (mean_Mon)
exponatial_Mon=np.append(results_Mon,exponential_smoothing(results_Mon,0.716)) #alpha is 0.8 or 0.9 something
#print (exponatial_Mon)
doble_exponatial_Mon=np.append(results_Mon,double_exponential_smoothing(results_Mon,0.9,0.45))
#print(doble_exponatial_Mon)
triple_exponential_Mon=triple_exponential_smoothing(np_result_Mon,12,0.716,0.0029,0.993,5)
print (triple_exponential_Mon)
fig=plt.figure()
fig.suptitle('Inventory_Forcast_For_Monday',fontweight='bold')
#plt.plot(results_Mon,label='Monday',linestyle="-")
# plt.figure(2)
# plt.plot(avg_Mon,label='Avg_Mon',linestyle="dotted")
# plt.figure(3)
#plt.plot(mean_Mon,label='mean_Mon',linestyle=":")
# plt.figure(4)
#plt.plot(exponatial_Mon,label='expoental_Mon',linestyle=":")
plt.plot(results_Mon,label='Monday',linestyle="-")
plt.plot(triple_exponential_Mon,label='Halt_Winter_Monday_Model',linestyle=":")
plt.grid('on')
#plt.plot(doble_exponatial_Mon,label='double_expoental_Mon',linestyle=":")
plt.legend(loc='upper left')
plt.show()





