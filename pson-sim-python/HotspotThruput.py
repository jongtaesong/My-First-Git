# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:00:21 2014

@author: hp
"""

import string
import numpy as np
import matplotlib.pyplot as plt

from Tkinter import *
from tkFileDialog import *




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

def Parse_data2(STR):
    Temp1=STR.split('from ')
    dummy1=Temp1[1].split(' to ')
    a=int(dummy1[0])
    dummy2 =dummy1[1].split(' ingress ')
    b=int(dummy2[0])
    dummy3 = dummy2[1].split(' egress ')
    c=int(dummy3[0])
    d=int(dummy3[1])
    #    return a,b,c,d
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
   
def MAKE_number_ARRAY2(DATA):
    A=[]
    B=[]
    C=[]
    D=[]
    for dat in DATA:
        if dat=='':
            break        
        a,b,c,d=Parse_data2(dat)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)        
    return A,B,C,D

def column(matrix, i):
    return [row[i] for row in matrix]
    
#filename=FILENAME()
test_hs_list = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44]
test_iter_list = [1,2,3,4,5]

'''
test_load_list = [85,90,95]
test_iter_list = [1,2,3]
'''

temp_average_delay=[]
temp_percentile_delay_90=[]
temp_percentile_delay_99=[]

gapa_average_delay=[[]]
gapa_percentile_delay_90=[[]]
gapa_percentile_delay_99=[[]]

temp_gapa_average_delay=[]
temp_gapa_percentile_delay_90=[]
temp_gapa_percentile_delay_99=[]

ga_average_delay=[[]]
ga_percentile_delay_90=[[]]
ga_percentile_delay_99=[[]]

temp_ga_average_delay=[]
temp_ga_percentile_delay_90=[]
temp_ga_percentile_delay_99=[]


Delay_threshold = 5500
Que_threshold = 5
Switch_size = 16
TrafficLoad=100


basic_thruput_hs=[]
basic_thruput=[]
ga_thruput_hs=[]
ga_thruput=[]
gapa_thruput_hs=[]
gapa_thruput=[]
hs_load_list=[]
for i in range(0,Switch_size):
    hs_load_list.append((float(i)+1.0)*100.0/float(Switch_size))
    
import matplotlib.pyplot
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10.5, 3.5)
matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    
# print basic iterations
for k in range(1,2):
    for j in range(0,len(test_iter_list)):  
        for i in range(0,Switch_size):
        
  
            #print('speed up, iteration, hot shot factor',i,j,k)
            #print(test_load_index)
            #print(test_iter_index)
            filename="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/sim_result_truput_D10000_S"
            filename+=str(Switch_size)
            filename+="_L"
            filename+=str(TrafficLoad)
            filename+="_I"
            filename+=str(test_iter_list[j])
            filename+="_Tth-1_Qth-1_GA0_HS"
            filename+=str(i)
            filename+="_SU"
            filename+=str(k)
            filename+=".txt"
            
            DATA = FILE_TO_ARRAY(filename)    
            basicFrom,basicTo,basicIn,basicOut = MAKE_number_ARRAY2(DATA)
            accum_in = 0
            accum_out = 0
            accum_in_hs = 0
            accum_out_hs = 0
            for ii in range(0,len(basicFrom)):
                if basicFrom[ii]==basicTo[ii]:
                    accum_in_hs += basicIn[ii]
                    accum_out_hs += basicOut[ii]
                accum_in += basicIn[ii]
                accum_out += basicOut[ii]
         
            thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
            basic_thruput_hs.append(thruput_hs)
            thruput = float(accum_out)*100.0/float(accum_in)
            basic_thruput.append(thruput)
                
    
            
            

        #for i in range(0,5):
        print(basic_thruput_hs)
        plt.subplot(1,2,1)
        hotspot = plt.plot(hs_load_list,basic_thruput_hs)
        basic_thruput_hs=[]

        plt.subplot(1,2,2)
        overall = plt.plot(hs_load_list,basic_thruput)
        basic_thruput=[]

    plt.subplot(1,2,1)
    plt.title('Hotspot throughput BASIC - switch:'+str(Switch_size), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14,)
    plt.ylim(10,100)
    plt.xlim(0,100)
    plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'))


    plt.subplot(1,2,2)
    plt.title('Overall throughput BASIC - switch:'+str(Switch_size), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14)
    plt.ylim(10,100)
    plt.xlim(0,100) 
    plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'))       
    plt.show()  
        
    #matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    
    #plt.show()
    #overall.show()
        
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10.5, 3.5) 
# print grant aware iterations
for k in range(1,2):
    for j in range(0,len(test_iter_list)):  
        for i in range(0,Switch_size):
        
  
            #print('speed up, iteration, hot shot factor',i,j,k)
            #print(test_load_index)
            #print(test_iter_index)
            filename="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/sim_result_truput_D10000_S"
            filename+=str(Switch_size)
            filename+="_L"
            filename+=str(TrafficLoad)
            filename+="_I"
            filename+=str(test_iter_list[j])
            filename+="_Tth-1_Qth-1_GA1_HS"
            filename+=str(i)
            filename+="_SU"
            filename+=str(k)
            filename+=".txt"
            
            DATA = FILE_TO_ARRAY(filename)    
            gaFrom,gaTo,gaIn,gaOut = MAKE_number_ARRAY2(DATA)
            accum_in = 0
            accum_out = 0
            accum_in_hs = 0
            accum_out_hs = 0
            for ii in range(0,len(gaFrom)):
                if gaFrom[ii]==gaTo[ii]:
                    accum_in_hs += gaIn[ii]
                    accum_out_hs += gaOut[ii]
            
                accum_in += gaIn[ii]
                accum_out += gaOut[ii]
         
            thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
            ga_thruput_hs.append(thruput_hs)
            thruput = float(accum_out)*100.0/float(accum_in)
            ga_thruput.append(thruput)
                
    
            
            
        print(ga_thruput_hs)

        #for i in range(0,5):
        plt.subplot(1,2,1)
        hotspot = plt.plot(hs_load_list,ga_thruput_hs)
        ga_thruput_hs=[]
        plt.subplot(1,2,2)
        overall = plt.plot(hs_load_list,ga_thruput)
        ga_thruput=[]
        

    #params = {'legend.fontsize': 20, 'legend.linewidth': 1}
    plt.subplot(1,2,1)
    plt.title('Hotspot throughput GA- switch:'+str(Switch_size), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14)
    plt.ylim(10,100)
    plt.xlim(0,100)
    #matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'))
    #plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'), loc='lower right',frameon=False,fontsize=10, linewidth=1.5)
    '''
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
    plt.setp(ltext, fontsize='large')    # the legend text fontsize
    plt.setp(llines, linewidth=1.5)      # the legend linewidth
    '''
    
    plt.subplot(1,2,2)
    plt.title('Overall throughput GA - switch:'+str(Switch_size), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14)
    plt.ylim(10,100)
    plt.xlim(0,100)
    #matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'))
    plt.show()






fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10.5, 3.5) 
# print grant + priority aware iterations
for k in range(1,2):
    for j in range(0,len(test_iter_list)):  
        for i in range(0,Switch_size):
        
            filename="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/sim_result_truput_D100000_S"
            filename+=str(Switch_size)
            filename+="_L"
            filename+=str(TrafficLoad)
            filename+="_I"
            filename+=str(test_iter_list[j])
            filename+="_Tth"
            filename+=str(Delay_threshold)
            filename+="_Qth"
            filename+=str(Que_threshold)
            filename+="_GA4_HS"
            filename+=str(i)
            filename+="_SU"
            filename+=str(k)
            filename+=".txt"
            
            DATA = FILE_TO_ARRAY(filename)    
            gapaFrom,gapaTo,gapaIn,gapaOut = MAKE_number_ARRAY2(DATA)
            accum_in = 0
            accum_out = 0
            accum_in_hs = 0
            accum_out_hs = 0
            for ii in range(0,len(gapaFrom)):
                if gapaFrom[ii]==gapaTo[ii]:
                    accum_in_hs += gapaIn[ii]
                    accum_out_hs += gapaOut[ii]
            
                accum_in += gapaIn[ii]
                accum_out += gapaOut[ii]
         
            thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
            gapa_thruput_hs.append(thruput_hs)
            thruput = float(accum_out)*100.0/float(accum_in)
            gapa_thruput.append(thruput)
                
    
            
            

        #for i in range(0,5):
        print(gapa_thruput_hs)
        
        plt.subplot(1,2,1)
        hotspot = plt.plot(hs_load_list,gapa_thruput_hs)
        gapa_thruput_hs=[]
        plt.subplot(1,2,2)
        overall = plt.plot(hs_load_list,gapa_thruput)
        gapa_thruput=[]

    #params = {'legend.fontsize': 20, 'legend.linewidth': 1}
    plt.subplot(1,2,1)
    plt.title('Hotspot throughput GAPA - switch:'+str(Switch_size), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14)
    plt.ylim(10,100)
    plt.xlim(0,100)
    #matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'))
    #plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'), loc='lower right',frameon=False,fontsize=10, linewidth=1.5)
    '''
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
    plt.setp(ltext, fontsize='large')    # the legend text fontsize
    plt.setp(llines, linewidth=1.5)      # the legend linewidth
    '''
    
    plt.subplot(1,2,2)
    plt.title('Overall throughput GAPA - switch:'+str(Switch_size), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14)
    plt.ylim(60,100)
    plt.xlim(0,100)
    #matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    plt.legend(('iter:1','iter:2','iter:3','iter:4','iter:5'))
    plt.show()
    