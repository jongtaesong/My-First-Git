# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:00:21 2014

@author: hp
"""

import string
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats

from Tkinter import *
from tkFileDialog import *
import matplotlib.pyplot



'''
def FILENAME():
    root = Tk() 
    root.withdraw()
    return askopenfilename(parent=root)
'''
def FILE_TO_ARRAY(filename):
    f = open(filename,'r')
    string=f.read()
    f.close()
    return string.split('\n')      
    
def Parse_data(STR):
    Temp=STR.split(',')
    dummy=Temp[0].split('from ')
    A=dummy[1].split(' to ')
    a=int(A[0])
    b=int(A[1])
    
    dummy=Temp[1].split('delay = ')    
    B=dummy[1].split(' at timeslot ')
    c=int(B[0])
    d=int(B[1])
    return a,b,c,d

def MAKE_number_ARRAY(DATA):
    A=[]
    B=[]
    C=[]
    D=[]
    for dat in DATA:
        if dat=='':
            break        
        a,b,c,d=Parse_data(dat)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)        
    return A,B,C,D
   
def column(matrix, i):
    return [row[i] for row in matrix]
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
    
    

def make_filenamelqf(duration, switch_size, load, iteration, Tth, Qth, ganumber):
    a="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/sim_result_D"
    a+=str(duration)
    a+="_S"
    a+=str(switch_size)
    a+="_L"
    a+=str(load)
    a+="_I"
    a+=str(iteration)
    a+="_Tth"
    a+=str(Tth)
    a+="_Qth"
    a+=str(Qth)
    a+="_GA"
    a+=str(ganumber)
    a+=".txt"
    return a
    
    
    
def make_filename(duration, switch_size, load, iteration, Tth, Qth, ganumber):
    a="d:/2017/rhapsody/PSONScheduler/component_4/DefaultConfig/sim_result_D"
    a+=str(duration)
    a+="_S"
    a+=str(switch_size)
    a+="_L"
    a+=str(load)
    a+="_I"
    a+=str(iteration)
    a+="_Tth"
    a+=str(Tth)
    a+="_Qth"
    a+=str(Qth)
    a+="_GA"
    a+=str(ganumber)
    a+=".txt"
    return a
    
    
    
    
def plot_delay_graph (load_list, iter_list, average_delay, schemename):
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    for i in range(0,len(iter_list)):
        plt.semilogy(load_list, average_delay[i][:])
        #fig = matplotlib.pyplot.gcf()
        #fig.set_size_inches(4.5, 3.5)
        #matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')    
            
    plt.title( schemename)
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    #plt.ylim(0,1000)
    plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
    plt.show()





    
#filename=FILENAME()
test_load_list = [50, 52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96, 98, 99]
test_iter_list = [1,2,3,4,5]
#test_load_list = [96]
#test_iter_list = [3]

'''
test_load_list = [85,90,95]
test_iter_list = [1,2,3]
'''


filename="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/sim_result_D10000_S"
filename+=str(Switch_size)
filename+="_L"
filename+=str(test_load_list[i])
filename+="_I"
filename+=str(test_iter_list[j])
filename+="_Tth0_Qth0_GA3.txt"
        

basic_average_delay=[[]]
basic_average_hi_95=[[]]
basic_average_lo_95=[[]]
basic_percentile_delay_90=[[]]
basic_percentile_delay_99=[[]]

temp_basic_average_delay=[]
temp_basic_average_hi_95=[]
temp_basic_average_lo_95=[]
temp_basic_percentile_delay_90=[]
temp_basic_percentile_delay_99=[]

ga_average_delay=[[]]
ga_average_hi_95=[[]]
ga_average_lo_95=[[]]
ga_percentile_delay_90=[[]]
ga_percentile_delay_99=[[]]

temp_ga_average_delay=[]
temp_ga_average_hi_95=[]
temp_ga_average_lo_95=[]
temp_ga_percentile_delay_90=[]
temp_ga_percentile_delay_99=[]

clqf_average_delay=[[]]
clqf_average_hi_95=[[]]
clqf_average_lo_95=[[]]
clqf_percentile_delay_90=[[]]
clqf_percentile_delay_99=[[]]

temp_clqf_average_delay=[]
temp_clqf_average_hi_95=[]
temp_clqf_average_lo_95=[]
temp_clqf_percentile_delay_90=[]
temp_clqf_percentile_delay_99=[]


gapa_average_delay=[[]]
gapa_average_hi_95=[[]]
gapa_average_lo_95=[[]]
gapa_percentile_delay_90=[[]]
gapa_percentile_delay_99=[[]]

temp_gapa_average_delay=[]
temp_gapa_average_hi_95=[]
temp_gapa_average_lo_95=[]
temp_gapa_percentile_delay_90=[]
temp_gapa_percentile_delay_99=[]

Delay_threshold = 600
Que_threshold = 5
Switch_size = 30
Duration = 10000


'''
for i in range(0,3):
    for j in range(0,3):    
'''
for j in range(0,len(test_iter_list)):  
    for i in range(0,len(test_load_list)):
  
        print(i,j)

        filename=make_filenamelqf(Duration, Switch_size, test_load_list[i], test_iter_list[j],0, 0, 4)
        DATA = FILE_TO_ARRAY(filename)    
        clqfA,clqfB,clqfC,clqfD = MAKE_number_ARRAY(DATA)
        sample_mean, sample_hi_95, sample_lo_95 = mean_confidence_interval(clqfC, 0.98)
        temp_clqf_average_delay.append(sample_mean)
        temp_clqf_average_hi_95.append(sample_hi_95)
        temp_clqf_average_lo_95.append(sample_lo_95)
        temp_clqf_percentile_delay_90.append(np.percentile(clqfC, 90))
        temp_clqf_percentile_delay_99.append(np.percentile(clqfC, 99))

        filename = make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j], -1, -1, 0)     
        DATA = FILE_TO_ARRAY(filename)    
        basicA,basicB,basicC,basicD = MAKE_number_ARRAY(DATA)
        sample_mean, sample_hi_95, sample_lo_95 = mean_confidence_interval(basicC, 0.98)
        temp_basic_average_delay.append(sample_mean)
        temp_basic_average_hi_95.append(sample_hi_95)
        temp_basic_average_lo_95.append(sample_lo_95)
        temp_basic_percentile_delay_90.append(np.percentile(basicC, 90))
        temp_basic_percentile_delay_99.append(np.percentile(basicC, 99))
    
        filename=make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j],-1, -1, 1)
        DATA = FILE_TO_ARRAY(filename)    
        gaA,gaB,gaC,gaD = MAKE_number_ARRAY(DATA)
        sample_mean, sample_hi_95, sample_lo_95 = mean_confidence_interval(gaC, 0.98)
        temp_ga_average_delay.append(sample_mean)
        temp_ga_average_hi_95.append(sample_hi_95)
        temp_ga_average_lo_95.append(sample_lo_95)
        temp_ga_percentile_delay_90.append(np.percentile(gaC, 90))
        temp_ga_percentile_delay_99.append(np.percentile(gaC, 99))
    
        filename=make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j],Delay_threshold, Que_threshold, 1)
        DATA = FILE_TO_ARRAY(filename)    
        gapaA,gapaB,gapaC,gapaD = MAKE_number_ARRAY(DATA)
        sample_mean, sample_hi_95, sample_lo_95 = mean_confidence_interval(gapaC, 0.98)
        temp_gapa_average_delay.append(sample_mean)
        temp_gapa_average_hi_95.append(sample_hi_95)
        temp_gapa_average_lo_95.append(sample_lo_95)
        temp_gapa_percentile_delay_90.append(np.percentile(gapaC, 90))
        temp_gapa_percentile_delay_99.append(np.percentile(gapaC, 99))
        
        
        
        

        
    if j==0:

    
        basic_average_delay[j]=temp_basic_average_delay
        basic_average_hi_95[j]=temp_basic_average_hi_95
        basic_average_lo_95[j]=temp_basic_average_lo_95
        basic_percentile_delay_90[j]=temp_basic_percentile_delay_90
        basic_percentile_delay_99[j]=temp_basic_percentile_delay_99
        
        clqf_average_delay[j]=temp_clqf_average_delay
        clqf_average_hi_95[j]=temp_clqf_average_hi_95
        clqf_average_lo_95[j]=temp_clqf_average_lo_95
        clqf_percentile_delay_90[j]=temp_clqf_percentile_delay_90
        clqf_percentile_delay_99[j]=temp_clqf_percentile_delay_99
        
        ga_average_delay[j]=temp_ga_average_delay
        ga_average_hi_95[j]=temp_ga_average_hi_95
        ga_average_lo_95[j]=temp_ga_average_lo_95
        ga_percentile_delay_90[j]=temp_ga_percentile_delay_90
        ga_percentile_delay_99[j]=temp_ga_percentile_delay_99
        
        gapa_average_delay[j]=temp_gapa_average_delay
        gapa_average_hi_95[j]=temp_gapa_average_hi_95
        gapa_average_lo_95[j]=temp_gapa_average_lo_95
        gapa_percentile_delay_90[j]=temp_gapa_percentile_delay_90
        gapa_percentile_delay_99[j]=temp_gapa_percentile_delay_99
        
    else:

        basic_average_delay.append(temp_basic_average_delay)
        basic_average_hi_95.append(temp_basic_average_hi_95)
        basic_average_lo_95.append(temp_basic_average_lo_95)
        basic_percentile_delay_90.append(temp_basic_percentile_delay_90)
        basic_percentile_delay_99.append(temp_basic_percentile_delay_99)
        
        clqf_average_delay.append(temp_clqf_average_delay)
        clqf_average_hi_95.append(temp_clqf_average_hi_95)
        clqf_average_lo_95.append(temp_clqf_average_lo_95)
        clqf_percentile_delay_90.append(temp_clqf_percentile_delay_90)
        clqf_percentile_delay_99.append(temp_clqf_percentile_delay_99)
        
        ga_average_delay.append(temp_ga_average_delay)
        ga_average_hi_95.append(temp_ga_average_hi_95)
        ga_average_lo_95.append(temp_ga_average_lo_95)
        ga_percentile_delay_90.append(temp_ga_percentile_delay_90)
        ga_percentile_delay_99.append(temp_ga_percentile_delay_99)
        
        gapa_average_delay.append(temp_gapa_average_delay)
        gapa_average_hi_95.append(temp_gapa_average_hi_95)
        gapa_average_lo_95.append(temp_gapa_average_lo_95)
        gapa_percentile_delay_90.append(temp_gapa_percentile_delay_90)
        gapa_percentile_delay_99.append(temp_gapa_percentile_delay_99)
        


    


    temp_basic_average_delay=[]
    temp_basic_average_hi_95=[]
    temp_basic_average_lo_95=[]
    temp_basic_percentile_delay_90=[]
    temp_basic_percentile_delay_99=[]

    temp_clqf_average_delay=[]
    temp_clqf_average_hi_95=[]
    temp_clqf_average_lo_95=[]
    temp_clqf_percentile_delay_90=[]
    temp_clqf_percentile_delay_99=[]

    temp_ga_average_delay=[]
    temp_ga_average_hi_95=[]
    temp_ga_average_lo_95=[]
    temp_ga_percentile_delay_90=[]
    temp_ga_percentile_delay_99=[]

    temp_gapa_average_delay=[]
    temp_gapa_average_hi_95=[]
    temp_gapa_average_lo_95=[]
    temp_gapa_percentile_delay_90=[]
    temp_gapa_percentile_delay_99=[]


    #for i in range(0,5):
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')

    plt.semilogy(test_load_list,basic_average_delay[j][:])
    plt.semilogy(test_load_list,ga_average_delay[j][:])
    plt.semilogy(test_load_list,gapa_average_delay[j][:])
    plt.semilogy(test_load_list,clqf_average_delay[j][:])

    plt.title('average delay iteration:'+str(test_iter_list[j])+' switch:'+str(Switch_size))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('Basic','GA','GAPA','C-LQF'), loc='upper left')
    plt.show()
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    plt.semilogy(test_load_list,basic_percentile_delay_99[j][:])
    plt.semilogy(test_load_list,clqf_percentile_delay_99[j][:])
    plt.semilogy(test_load_list,ga_percentile_delay_99[j][:])
    plt.semilogy(test_load_list,gapa_percentile_delay_99[j][:])

    plt.title('99% delay iteration:'+str(test_iter_list[j])+' switch:'+str(Switch_size))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('Basic','GA','GAPA','C-LQF'), loc='upper left')
    plt.show()


plot_delay_graph (test_load_list, test_iter_list, basic_average_delay, "Average Delay - Basic")
plot_delay_graph (test_load_list, test_iter_list, clqf_average_delay, "Average Delay - C-LQF")
plot_delay_graph (test_load_list, test_iter_list, ga_average_delay, "Average Delay - GA")
plot_delay_graph (test_load_list, test_iter_list, gapa_average_delay, "Average Delay - GAPA")


plot_delay_graph (test_load_list, test_iter_list, basic_percentile_delay_99, "99% Delay - Basic")
plot_delay_graph (test_load_list, test_iter_list, clqf_percentile_delay_99, "99% Delay - C-LQF")
plot_delay_graph (test_load_list, test_iter_list, ga_percentile_delay_99, "99% Delay - GA")
plot_delay_graph (test_load_list, test_iter_list, gapa_percentile_delay_99, "99% Delay - GAPA")



for j in range(0,len(test_iter_list)): 
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')

    #plt.semilogy(test_load_list,basic_average_delay[j][:])
    plt.semilogy(test_load_list,ga_average_delay[j][:])
    plt.semilogy(test_load_list,gapa_average_delay[j][:])
    plt.semilogy(test_load_list,clqf_average_delay[j][:])

    plt.title('average delay iteration:'+str(test_iter_list[j])+' switch:'+str(Switch_size))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    #plt.legend(('Basic','GA','GAPA','C-LQF'), loc='upper left')
    plt.legend(('GA','GAPA','C-LQF'), loc='upper left')
    plt.show()


for j in range(0,len(test_iter_list)): 
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')

    #plt.semilogy(test_load_list,basic_average_delay[j][:])
    plt.plot(test_load_list,ga_average_delay[j][:])
    plt.plot(test_load_list,gapa_average_delay[j][:])
    plt.plot(test_load_list,clqf_average_delay[j][:])

    plt.title('average delay iteration:'+str(test_iter_list[j])+' switch:'+str(Switch_size))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    #plt.legend(('Basic','GA','GAPA','C-LQF'), loc='upper left')
    plt.legend(('GA','GAPA','C-LQF'), loc='upper left')
    plt.show()






for i in range(0,len(test_iter_list)):
    plt.semilogy(test_load_list, basic_percentile_delay_99[i][:])
    
plt.title('99% delay - basic')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
plt.show()


for i in range(0,len(test_iter_list)):
    plt.semilogy(test_load_list, clqf_percentile_delay_99[i][:])
    
plt.title('99% delay - LQF ')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
plt.show()



for i in range(0,len(test_iter_list)):
    plt.semilogy(test_load_list, basic_average_delay[i][:])
    plt.semilogy(test_load_list, clqf_average_delay[i][:])
    plt.title('average delay - iteration:'+str(i))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('basic', 'LQF'),loc='upper left')
    plt.show()
    
    
for i in range(0,len(test_iter_list)):
    #plt.semilogy(test_load_list, percentile_delay_99[i][:])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6.5, 5.5)
    matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')    
    plt.semilogy(test_load_list, basic_percentile_delay_99[i][:])
    plt.semilogy(test_load_list, clqf_percentile_delay_99[i][:])
    plt.title('99% delay')
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.xlim(80,100)
    #plt.legend(('basic','grant aware', 'grant+priority aware'),loc='upper left')
    plt.legend(('basic - iteration:'+str(i),'LQF - iteration:'+str(i)),loc='upper left')

plt.show()


for i in range(1,5):
    #plt.semilogy(test_load_list, percentile_delay_99[i][:])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6.5, 5.5)
    matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')    
    plt.semilogy(test_load_list, basic_percentile_delay_99[i][:])
    plt.semilogy(test_load_list, clqf_percentile_delay_99[i][:])

    #plt.legend(('basic','grant aware', 'grant+priority aware'),loc='upper left')
    plt.legend(('basic - iteration:'+str(i),'LQF - iteration:'+str(i)),loc='upper left')

plt.legend(('basic - iteration:1','LQF - iteration:1','basic - iteration:2','LQF - iteration:2'),loc='upper left')
plt.title('99% delay')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.xlim(80,100)
plt.show()

for i in range(1,5):
    #plt.semilogy(test_load_list, percentile_delay_99[i][:])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(3.5, 2.5)
    matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')    
    plt.plot(test_load_list, basic_percentile_delay_99[i][:],'--')
    plt.plot(test_load_list, clqf_percentile_delay_99[i][:])

    #plt.legend(('basic','grant aware', 'grant+priority aware'),loc='upper left')
    plt.legend(('basic','LQF'),loc='upper left')

    #plt.legend(('grant aware - iteration:3','grant+priority aware - iteration:3','grant aware - iteration:4','grant+priority aware - iteration:4'),loc='upper left')
    plt.title('99% delay  - iteration:'+str(i))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.xlim(80,100)
    plt.ylim(10,2000)
    plt.show()
    
    
    
for i in range(0,len(test_iter_list)):
    plt.semilogy(test_load_list, basic_percentile_delay_90[i][:])
    plt.semilogy(test_load_list, clqf_percentile_delay_90[i][:])
    plt.title('90% delay - iteration:'+str(i))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('basic','LQF'),loc='upper left')
    plt.show()






fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10.5, 5.5)
matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
#plt.plot(test_load_list, percentile_delay_99[1][:], label='1 iter-basic')
plt.semilogy(test_load_list, basic_percentile_delay_99[1][:],label='1 iter-basic')
plt.semilogy(test_load_list, clqf_percentile_delay_99[1][:],label='1 iter-LQF')
#plt.plot(test_load_list, percentile_delay_99[2][:], label='2 iter-basic')
plt.semilogy(test_load_list, basic_percentile_delay_99[2][:],label='2 iter-basic')
plt.semilogy(test_load_list, clqf_percentile_delay_99[2][:],label='2 iter-LQF')
#plt.plot(test_load_list, percentile_delay_99[3][:], label='3 iter-basic')
plt.semilogy(test_load_list, basic_percentile_delay_99[3][:],label='3 iter-basic')
plt.semilogy(test_load_list, clqf_percentile_delay_99[3][:],label='3 iter-LQF')
#plt.plot(test_load_list, percentile_delay_99[4][:], label='4 iter-basic')
plt.semilogy(test_load_list, basic_percentile_delay_99[4][:],label='4 iter-basic')
plt.semilogy(test_load_list, clqf_percentile_delay_99[4][:],label='4 iter-LQF')
plt.title('99% delay - iteration:'+str(i))
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(loc='upper left')
plt.show()






import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines



labels =['lqf','iter']
colors = ['r','b']
markers = ['*','--']

lqf = plt.hist(lqfC, 50, normed=1, histtype='step', cumulative=False, log=True )

basic = plt.hist(basicC, 1000, normed=1, histtype='step', cumulative=-1, color = 'g', log=True )
lqf = plt.hist(lqfC, 1000, normed=1, histtype='step', cumulative=-1, color = 'b' , alpha = 0.5, log=True)
plt.title('Delay Distribution')
plt.xlabel('Delay (time slots)')
plt.ylabel('Prob(delay > x)')
plt.xlim(3,4000)
plt.ylim(0.0001,1)
plt.legend(('GrantAware', 'Grant+Priority Aware'), loc='upper right')
plt.show()






lqf = plt.hist(lqfC, 100, normed=1, histtype='step', cumulative=-1, color = 'r', log=True )
basic = plt.hist(basicC, 100, normed=1, histtype='step', cumulative=-1, color = 'b', log=True )
#ga = plt.hist(gaC, 100, normed=1, histtype='step', cumulative=-1, color = 'purple', log=True )
plt.title('Tail Distribution')
plt.xlabel('Delay (time slots)')
plt.ylabel('Prob')
plt.legend(('LQf','basic'), loc='lower right')
#plt.legend([basic])
plt.show()

