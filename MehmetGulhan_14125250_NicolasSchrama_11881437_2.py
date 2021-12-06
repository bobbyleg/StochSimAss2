import numpy as np 
import random
import simpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
import seaborn as sns

lambd = 0.95 * 15  # mean inter-arrival time
mu = 1 * 15        # mean service time

n_experiments = 75
wait_array = np.empty(0) 
loop_var =  'Capacity'  # 'New customers' 

if loop_var == 'Capacity':
    new_customers_array = 10000                 # number of customers
    capacity_array = np.array((1, 2, 4))        # number of servers
else:
    new_customers_array = np.array((10, 50, 100, 1000))
    capacity_array = 1  

"""
The code for the source, customer, and queue functions is not originally ours. 
It is based on a code example from the simpy documentation. It can be found at:
https://simpy.readthedocs.io/en/latest/examples/bank_renege.html
Note that we significantly modified the original code. 
"""

# randomly generate customers
def source(env, customer_amount, counter, lambd, mu, method):
    for i in range(customer_amount):
        if method == 'm_d_n':
            time_served = 1/mu
        elif method == 'fat tail':
            draw = np.random.uniform(0,1)
            if draw < 0.25:
                time_served = random.expovariate(2/(5*mu)) 
            else:
                time_served = random.expovariate(2/mu) 
        else:
            time_served = random.expovariate(1/mu)
        c = customer(env, 'Customer%02d' % i, counter, time_served, method)
        env.process(c)
        t = random.expovariate(lambd)
        yield env.timeout(t)

# customer arrives, is served, and leaves
def customer(env, name, counter, time_served, method):
    global wait_array
    arrive = env.now
    #print(f'%7.4f %s: Here I am with time {time_served}' % (arrive, name))

    if method == 'priority':
        with counter.request(priority=time_served) as req:
            results = yield req 
            wait = env.now - arrive
            wait_array = np.append(wait_array, wait)

            #print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
            yield env.timeout(time_served)
            #print('%7.4f %s: Finished' % (env.now, name))

    else:
        with counter.request() as req:
            results = yield req 
            wait = env.now - arrive
            wait_array = np.append(wait_array, wait)

            #print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
            yield env.timeout(time_served)
            #print('%7.4f %s: Finished' % (env.now, name))

# Setup and start the simulation
def queue(customer_amount, capacity, lambd, mu, method):
    #print(f'Running a simulation with capacity = {capacity}.')
    env = simpy.Environment()
    if method == 'priority':
        counter = simpy.PriorityResource(env, capacity=capacity)
    else:
        counter = simpy.Resource(env, capacity=capacity)
    env.process(source(env, customer_amount, counter, lambd, mu, method))
    env.run()
    #print()
    return np.mean(wait_array)
    #print()
    #print()

# run a simulation for many capacities
def sim(non_loop_number, loop_array, n_experiments, lambd, mu, loop_var, method):
    global wait_array
    waiting_times = np.ones((len(loop_array), n_experiments))
    count = 0
    
    for element in tqdm(loop_array):
        waiting_times_per = np.empty(n_experiments)
        
        for experiment in range(n_experiments):
            if loop_var == 'Capacity':
                waiting_times_per[experiment] = queue(non_loop_number, element, lambd * element, mu, method) 
                wait_array = np.empty(0) 
            else:
                waiting_times_per[experiment] = queue(element, non_loop_number, lambd, mu, 'm_m_n') 
                wait_array = np.empty(0) 
            
        waiting_times[count] = np.array(waiting_times_per)
        count += 1

    return waiting_times

# this function returns p-values comparing average waiting time distributions for one queue method, e.g., M/M/n or M/D/n, for different numbers of servers
def p_value_dict(cap_list, waiting_times):
    p_vals_dict = {}

    for i in range(len(cap_list)):
        for j in range(i+1, len(cap_list)):
            p_vals_dict[str(cap_list[i]) + '_' + str(cap_list[j])] = stats.ttest_ind(waiting_times[i], waiting_times[j]).pvalue 
    
    return p_vals_dict

# this function returns p-values comparing average waiting time distributions for multiple queue methods, e.g., M/M/n and M/D/n, for a specific number of servers
def p_value_dict_two_methods(cap_list, waiting_times, mmn, method):
    p_vals_dict = {}

    for i in range(len(cap_list)):
        p_vals_dict[method + '_' + str(cap_list[i])] = stats.ttest_ind(waiting_times[i], mmn[i]).pvalue 
    
    return p_vals_dict

# similar too p_value_dict_two_methods function, except all but one capacities are used
# we use this function to compare priority M/M/n with 1 (2) server(s) to FIFO M/M/n with 2 (4) servers
def p_value_dict_priority(cap_list, waiting_times, mmn, method):
    p_vals_dict = {}

    for i in range(0, len(cap_list)-1):
        p_vals_dict[method + '_' + str(cap_list[i])] = stats.ttest_ind(waiting_times[i], mmn[i]).pvalue 
    
    return p_vals_dict

# boxplots
def plot(array, waiting_times, mmm_waiting_times, n_experiments, loop_var, method):
    dict = {}

    # for index, capacity in enumerate(waiting_times):
    #     dict[array[index]] = capacity
    for index, capacity in enumerate(waiting_times):
        dict[method + '_' + str(array[index])] = capacity
    if method != 'm_m_n':
        for index, capacity in enumerate(mmm_waiting_times):
            dict['m_m_n' + '_' + str(array[index])] = capacity

    fig, ax = plt.subplots()
    ax.boxplot(dict.values())
    ax.set_xticklabels(dict.keys())
    plt.xlabel(f"{loop_var}")
    plt.ylabel("Average waiting time")
    plt.title(f"{method}")
    plt.savefig(f"images/boxplots/boxplot_average_waiting_times_{method}_{n_experiments}.png")
    plt.show()

# Function calls
def all_methods(non_loop_number, loop_array, n_experiments, lambd, mu, loop_var, method_list):
    dict = {}
    p_vals_all_methods = {}
    p_vals_two_methods = {}

    # running 75 simulations per experiment
    for method in method_list:
        dict[method] = sim(new_customers_array, capacity_array, n_experiments, lambd, mu, loop_var, method)
    
    # checking if increasing capacity changes distribution per experiment
    for method in method_list:
        p_vals_all_methods[method] = p_value_dict(loop_array, dict[method])

    # checking per capacity if two methods are differently distributed
    for method in method_list[1:]:
        p_vals_two_methods[method] = p_value_dict_two_methods(loop_array, dict[method], dict['m_m_n'], method)

    # checking if priority is the same as doubling the number of servers in terms of distribution
    priority = p_value_dict_priority(loop_array, dict['priority'][:-1], dict['m_m_n'][1:], 'priority')
    
    # creating plots of all capacities per experiment
    for method in dict.items():
        plot(capacity_array, method[1], dict['m_m_n'], n_experiments, loop_var, method[0])
    
    
    return p_vals_all_methods, p_vals_two_methods, priority

p_value_dict, p_vals_two_methods, priority = all_methods(new_customers_array, capacity_array, n_experiments, lambd, mu, loop_var, ['m_m_n', 'priority', 'm_d_n','fat tail']) # 
print('checking if increasing capacity changes distribution per experiment')
print(p_value_dict)
print()
print('checking per capacity if two methods are differently distributed')
print(p_vals_two_methods)
print()
print('checking if priority is the same as doubling the number of servers in terms of distribution')
print(priority)


##################################### Creating plots to determine the right number of customers

customers_array= [100, 500, 1000, 10000, 50000, 100000]

# runs experiments used in the plots
def mmi_per_ncustomers(customers, lambd, mu):
    waiting_ = np.array([])
    for i in range(500):
        global wait_array
        wait_array = np.empty(0)
        waiting_ = np.append(waiting_, queue(customers, 1, lambd*1, mu, 'm_m_n'))
    return waiting_

# creates plots of the distributions
def plots_ncustomers(rho, waiting_, n_customers):
    colors = ['blue', 'yellow', 'red', 'green']
    for i in range(4): 
        sns.distplot(waiting_[i], hist=True, kde=True, 
                  bins=10, color = colors[i], 
                  hist_kws={'edgecolor':colors[i]},
                  kde_kws={'linewidth': 2},
                  label=str(n_customers[i]))
    plt.title("Distribution plots for different customer sizes, rho = " + str(rho))
    plt.legend()
    plt.show()
    return

# compares distributions
def distribution_check(customers_array):
    waiting_ = []
    for n_customers in customers_array[:4]:
        waiting_.append(mmi_per_ncustomers(n_customers, 0.1, 1))
    plots_ncustomers(0.1, waiting_, customers_array[:4])
    for i in range(len(waiting_)):
        print("rho = 0.1 -- p value normal test for n = ", customers_array[:4][i], " is ", stats.normaltest(waiting_[i]))
    
    waiting_2 = []
    for n_customers in customers_array[2:]:
        waiting_2.append(mmi_per_ncustomers(n_customers, 0.95, 1))
    plots_ncustomers(0.95, waiting_2, customers_array[2:])
    for i in range(len(waiting_2)):
        print("rho = 0.95 -- p value normal test for n = ", customers_array[2:][i], " is ", stats.normaltest(waiting_2[i]))
    return waiting_, waiting_2

distribution_check(customers_array)