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
    
#filename=FILENAME()
test_load_list = [52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96]
test_iter_list = [1,2,3,4,5]

'''
test_load_list = [85,90,95]
test_iter_list = [1,2,3]
'''

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

islip_average_delay=[[]]
islip_average_hi_95=[[]]
islip_average_lo_95=[[]]
islip_percentile_delay_90=[[]]
islip_percentile_delay_99=[[]]

temp_islip_average_delay=[]
temp_islip_average_hi_95=[]
temp_islip_average_lo_95=[]
temp_islip_percentile_delay_90=[]
temp_islip_percentile_delay_99=[]

Delay_threshold = 60
Que_threshold = 5
Switch_size = 30


'''
for i in range(0,3):
    for j in range(0,3):    
'''
for j in range(0,len(test_iter_list)):  
    for i in range(0,len(test_load_list)):
  
        print(i,j)
        #print(test_load_index)
        #print(test_iter_index)   
            
           
        filename="d:/2017/rhapsody/PSONScheduler/component_4/DefaultConfig/sim_result_D10000_S"
        filename+=str(Switch_size)
        filename+="_L"
        filename+=str(test_load_list[i])
        filename+="_I"
        filename+=str(test_iter_list[j])
        filename+="_Tth-1_Qth-1_GA1.txt"
        DATA = FILE_TO_ARRAY(filename)    
        gaA,gaB,gaC,gaD = MAKE_number_ARRAY(DATA)
        sample_mean, sample_hi_95, sample_lo_95 = mean_confidence_interval(gaC, 0.98)
        temp_ga_average_delay.append(sample_mean)
        temp_ga_average_hi_95.append(sample_hi_95)
        temp_ga_average_lo_95.append(sample_lo_95)
        temp_ga_percentile_delay_90.append(np.percentile(gaC, 90))
        temp_ga_percentile_delay_99.append(np.percentile(gaC, 99))
    
        filename="d:/2017/rhapsody/PSONSchedulerIslip/component_4/DefaultConfig/sim_result_D10000_S"
        filename+=str(Switch_size)
        filename+="_L"
        filename+=str(test_load_list[i])
        filename+="_I"
        filename+=str(test_iter_list[j])
        filename+="_Tth-1_Qth-1_GA0.txt"
        DATA = FILE_TO_ARRAY(filename)    
        islipA,islipB,islipC,islipD = MAKE_number_ARRAY(DATA)
        sample_mean, sample_hi_95, sample_lo_95 = mean_confidence_interval(islipC, 0.98)
        temp_islip_average_delay.append(sample_mean)
        temp_islip_average_hi_95.append(sample_hi_95)
        temp_islip_average_lo_95.append(sample_lo_95)
        temp_islip_percentile_delay_90.append(np.percentile(islipC, 90))
        temp_islip_percentile_delay_99.append(np.percentile(islipC, 99))
        
        '''     
        basic = plt.hist(basicC, 100, normed=1, histtype='step', cumulative=-1, color = 'r')
        noga = plt.hist(nogaC, 100, normed=1, histtype='step', cumulative=-1, color = 'b', log=True )
        ga = plt.hist(gaC, 100, normed=1, histtype='step', cumulative=-1, color = 'purple', log=True )
        plt.title('Tail Distribution iteration '+str(test_iter_list[j])+'  load '+str(test_load_list[i]))
        plt.xlabel('Delay (time slots)')
        plt.ylabel('Prob')
        plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
        #plt.legend([basic])
        plt.show()
        '''
        
    if j==0:
        average_delay[j]=temp_average_delay
        average_hi_95[j]=temp_average_hi_95
        average_lo_95[j]=temp_average_lo_95
        percentile_delay_90[j]=temp_percentile_delay_90
        percentile_delay_99[j]=temp_percentile_delay_99
    
        ga_average_delay[j]=temp_ga_average_delay
        ga_average_hi_95[j]=temp_ga_average_hi_95
        ga_average_lo_95[j]=temp_ga_average_lo_95
        ga_percentile_delay_90[j]=temp_ga_percentile_delay_90
        ga_percentile_delay_99[j]=temp_ga_percentile_delay_99
        
        islip_average_delay[j]=temp_islip_average_delay
        islip_average_hi_95[j]=temp_islip_average_hi_95
        islip_average_lo_95[j]=temp_islip_average_lo_95
        islip_percentile_delay_90[j]=temp_islip_percentile_delay_90
        islip_percentile_delay_99[j]=temp_islip_percentile_delay_99
        
    else:
        
        ga_average_delay.append(temp_ga_average_delay)
        ga_average_hi_95.append(temp_ga_average_hi_95)
        ga_average_lo_95.append(temp_ga_average_lo_95)
        ga_percentile_delay_90.append(temp_ga_percentile_delay_90)
        ga_percentile_delay_99.append(temp_ga_percentile_delay_99)
        
        islip_average_delay.append(temp_islip_average_delay)
        islip_average_hi_95.append(temp_islip_average_hi_95)
        islip_average_lo_95.append(temp_islip_average_lo_95)
        islip_percentile_delay_90.append(temp_islip_percentile_delay_90)
        islip_percentile_delay_99.append(temp_islip_percentile_delay_99)
        


    
    temp_ga_average_delay=[]
    temp_ga_average_hi_95=[]
    temp_ga_average_lo_95=[]
    temp_ga_percentile_delay_90=[]
    temp_ga_percentile_delay_99=[]

    temp_islip_average_delay=[]
    temp_islip_average_hi_95=[]
    temp_islip_average_lo_95=[]
    temp_islip_percentile_delay_90=[]
    temp_islip_percentile_delay_99=[]


    #for i in range(0,5):
    plt.plot(test_load_list,ga_average_delay[j][:])
    plt.plot(test_load_list,islip_average_delay[j][:])
    #plt.fill_between(test_load_list, average_hi_95[j][:], average_lo_95[j][:], color='gray', alpha=0.5)
    #plt.fill_between(test_load_list, ga_average_hi_95[j][:], ga_average_lo_95[j][:], color='blue', alpha=0.5)
    #plt.fill_between(test_load_list, islip_average_hi_95[j][:], islip_average_lo_95[j][:], color='red', alpha=0.5)
    #plt.plot(test_load_list,average_hi_95[j][:])
    #plt.plot(test_load_list,ga_average_hi_95[j][:])
    #plt.plot(test_load_list,islip_average_hi_95[j][:])
    #plt.plot(test_load_list,average_lo_95[j][:])
    #plt.plot(test_load_list,ga_average_lo_95[j][:])
    #plt.plot(test_load_list,islip_average_lo_95[j][:])
    
    plt.title('average delay iteration:'+str(test_iter_list[j])+' switch:'+str(Switch_size)+' Dth:'+str(Delay_threshold)+' Qth:'+str(Que_threshold))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    #plt.legend(('Basic','Grant Aware', 'Priority+Grant Aware', '98% confidence interval'), loc='upper left')
    plt.legend(('Grant Aware', 'iSLIP'), loc='upper left')
    #plt.legend(('Basic','Grant Aware'), loc='upper left')
    plt.show()


for i in range(0,len(test_iter_list)):
    plt.plot(test_load_list, ga_average_delay[i][:])
    
plt.title('average delay - grant aware')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
plt.show()


for i in range(0,len(test_iter_list)):
    plt.plot(test_load_list, islip_average_delay[i][:])
    
plt.title('average delay - iSLIP ')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
plt.show()




for i in range(0,len(test_iter_list)):
    plt.plot(test_load_list, ga_percentile_delay_99[i][:])
    
plt.title('99% delay - grant aware')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
plt.show()


for i in range(0,len(test_iter_list)):
    plt.plot(test_load_list, islip_percentile_delay_99[i][:])
    
plt.title('99% delay - iSLIP ')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('iter:1','iter:2', 'iter:3', 'iter:4', 'iter:5'), loc='upper left')
plt.show()



for i in range(0,len(test_iter_list)):
    plt.plot(test_load_list, ga_average_delay[i][:])
    plt.plot(test_load_list, islip_average_delay[i][:])
    plt.title('average delay - iteration:'+str(i))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('basic','grant aware', 'grant+priority aware'),loc='upper left')
    plt.show()
    
    
for i in range(0,len(test_iter_list)):
    #plt.plot(test_load_list, percentile_delay_99[i][:])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6.5, 5.5)
    matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')    
    plt.plot(test_load_list, ga_percentile_delay_99[i][:])
    plt.plot(test_load_list, islip_percentile_delay_99[i][:])
    plt.title('99% delay - iteration:'+str(i))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    #plt.legend(('basic','grant aware', 'grant+priority aware'),loc='upper left')
    plt.legend(('grant aware', 'grant+priority aware'),loc='upper left')
    plt.show()
    
    
for i in range(0,len(test_iter_list)):
    plt.plot(test_load_list, ga_percentile_delay_90[i][:])
    plt.plot(test_load_list, islip_percentile_delay_90[i][:])
    plt.title('90% delay - iteration:'+str(i))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('basic','grant aware', 'grant+priority aware'),loc='upper left')
    plt.show()



fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10.5, 5.5)
matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
#plt.plot(test_load_list, percentile_delay_99[1][:], label='1 iter-basic')
plt.plot(test_load_list, ga_percentile_delay_99[1][:],label='1 iter-GA')
plt.plot(test_load_list, islip_percentile_delay_99[1][:],label='1 iter-GPA')
#plt.plot(test_load_list, percentile_delay_99[2][:], label='2 iter-basic')
plt.plot(test_load_list, ga_percentile_delay_99[2][:],label='2 iter-GA')
plt.plot(test_load_list, islip_percentile_delay_99[2][:],label='2 iter-GPA')
#plt.plot(test_load_list, percentile_delay_99[3][:], label='3 iter-basic')
plt.plot(test_load_list, ga_percentile_delay_99[3][:],label='3 iter-GA')
plt.plot(test_load_list, islip_percentile_delay_99[3][:],label='3 iter-GPA')
#plt.plot(test_load_list, percentile_delay_99[4][:], label='4 iter-basic')
plt.plot(test_load_list, ga_percentile_delay_99[4][:],label='4 iter-GA')
plt.plot(test_load_list, islip_percentile_delay_99[4][:],label='4 iter-GPA')
plt.title('99% delay - iteration:'+str(i))
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(loc='upper left')
plt.show()


ga_cycles=[2, 4, 6,8,10]
islip_cycles=[3,6,9,12,15]
for i in range(0,len(test_load_list)):
    # for j in range(0,len(test_iter_list)):  
    plt.semilogy(ga_cycles, column(ga_average_delay, i),'*', label='GA')
    plt.semilogy(islip_cycles, column(islip_average_delay, i),'x', label='iSILP')
    plt.title('average delay - load:'+str(test_load_list[i]))
    plt.xlim(0,16)
    plt.xlabel('cycle')
    plt.ylabel('Delay (time slots)')
    plt.legend(loc='upper right')
    plt.show()
        




'''
plt.plot(column(average_delay,0))
plt.plot(column(no_ga_average_delay,0))
plt.plot(column(ga_average_delay,0))
plt.title('average delay iteration:1')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
plt.show()


plt.plot(column(average_delay,0))
plt.plot(column(no_ga_average_delay,0))
plt.plot(column(ga_average_delay,0))
plt.title('average delay iteration:1')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
plt.show()

plt.plot(column(average_delay,0))
plt.plot(column(no_ga_average_delay,0))
plt.plot(column(ga_average_delay,0))
plt.title('average delay iteration:1 Dth:'+str(Delay_threshold)+' Qth:'+str(Que_threshold))
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
plt.show()
'''  





'''

    
plt.plot(test_load_list, percentile_delay_99)
plt.title('99% delay basic')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.show()

plt.plot(test_load_list, no_ga_percentile_delay_99)    
plt.title('99% delay grant aware')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.show()

plt.plot(test_load_list, ga_percentile_delay_99)
plt.title('99% delay grant + priority aware')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.show()

plt.plot(test_load_list, average_delay)
plt.title('average delay basic')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.show()

plt.plot(test_load_list, no_ga_average_delay)    
plt.title('average delay grant aware')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.show()

plt.plot(test_load_list, ga_average_delay)
plt.title('average delay grant + priority aware Dth:'+str(Delay_threshold)+" Qth:"+str(Que_threshold))
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.show()
    
diff_delay=[[]]
diff_delay=[[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]]
diff_delay_99=[[]]
diff_delay_99=[[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]]
            

for i in range(0,3):
    for j in range(0,3):    
for i in range(0,15):
    for j in range(0,5):    
        print(i,j)
        diff_delay[i][j] = no_ga_average_delay[i][j]-ga_average_delay[i][j]
        diff_delay_99[i][j] = no_ga_percentile_delay_99[i][j]-ga_percentile_delay_99[i][j]

plt.plot (test_load_list, diff_delay)
plt.show()
'''
     
'''    
fig = plt.figure()
ax = fig.gca(projection='3d')
#X = np.arange(20, 100, 10)
#Y = np.arange(1,6,1)

X = [20,30,40,50,60,70,80,90]
Y = [1,2,3,4,5]
X, Y = np.meshgrid(X, Y)
Z = average_delay
surf = ax.plot_surface(Y, X, average_delay, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

Y = [20,30,40,50,60,70,80,90]
X = [1,2,3,4,5]
Z = average_delay


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


#mu = 200
#sigma = 25
#x = mu + sigma*np.random.randn(10000)

#fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 10))

#plt.hist(basicC, 50, normed=0, histtype='stepfilled', facecolor='g', alpha=0.5)
#plt.hist(iteronlyC, 50, normed=0, histtype='stepfilled', facecolor='r', alpha=0.5)
labels =['basic','iter']
colors = ['r','b']
markers = ['*','--']
#total = [basicC, iteronlyC]
#plt.legend(handles=[basicC,iteronlyC])
#[basicC,iteronlyC],['redzzzzz','bluezzzzzz'])
#plt.legend(handles=[basic,iteronly])


#total_plot = plt.hist(total, 1000, normed=1, histtype='step', cumulative=True, color = colors, label=labels, marker=markers)

basic = plt.hist(basicC, 50, normed=1, histtype='step', cumulative=False, color = 'r', log=True )

noga = plt.hist(nogaC, 50, normed=1, histtype='step', cumulative=False, color = 'b', log=True )
ga = plt.hist(gaC, 50, normed=1, histtype='step', cumulative=False, color = 'purple', log=True )
plt.title('Delay Distribution')
plt.xlabel('Delay (time slots)')
plt.ylabel('Cummulertive Prob')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
#plt.legend(('Without GrantAware', 'With GrantAware'), loc='lower right')
#plt.legend([basic])
plt.show()






basic = plt.hist(basicC, 100, normed=1, histtype='step', cumulative=-1, color = 'r', log=True )
noga = plt.hist(nogaC, 100, normed=1, histtype='step', cumulative=-1, color = 'b', log=True )
#ga = plt.hist(gaC, 100, normed=1, histtype='step', cumulative=-1, color = 'purple', log=True )
plt.title('Tail Distribution')
plt.xlabel('Delay (time slots)')
plt.ylabel('Prob')
plt.legend(('Basic','Without GrantAware', 'With GrantAware'), loc='lower right')
#plt.legend([basic])
plt.show()


# Create a histogram by providing the bin edges (unequally spaced).
'''
bins = [100, 150, 180, 195, 205, 220, 250, 300]
ax1.hist(x, C, normed=1, histtype='bar', rwidth=0.8)
ax1.set_title('unequal bins')

plt.tight_layout()
plt.show()
'''