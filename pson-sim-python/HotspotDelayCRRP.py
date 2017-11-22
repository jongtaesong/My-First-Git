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
    dummy2 =dummy1[1].split(' accum_delay ')
    b=int(dummy2[0])
    dummy3 = dummy2[1].split(' count ')
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
    
def make_filename(duration, switch_size, load, iteration, Tth, Qth, hotspot, ganumber, scale):
    a="d:/2017/rhapsody/PSONSchedulerCyclicRRP/component_4/DefaultConfig/hotspot_delay_D"
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
    a+="_HS"
    a+=str(hotspot)
    a+="_SU"
    a+=str(scale)
    a+=".txt"
    #print(a)
    return a
    
    
def make_filename2(duration, switch_size, load, iteration, Tth, Qth, hotspot, ganumber):
    a="d:/2017/rhapsody/PSONScheduler/component_4/DefaultConfig/hotspot_delay_D"
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
    a+="_HS"
    a+=str(hotspot)
    a+=".txt"
    #print(a)
    return a
    
def make_filename3(duration, switch_size, load, iteration, Tth, Qth, hotspot, ganumber):
    a="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/hotspot_delay_D"
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
    a+="_HS"
    a+=str(hotspot)
    a+=".txt"
    #print(a)
    return a
    
    
def plot_delay_graph (load_list, iter_list, average_delay, schemename):
    for i in range(0,len(iter_list)):
        plt.semilogy(load_list, ldf_average_delay[i][:])
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
test_hs_list = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44]
test_iter_list = [1,2,3]


test_load_list = [52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96]
'''
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


Delay_threshold = 0
Que_threshold = 0
Switch_size = 20
TrafficLoad=100
Duration = 100000
ScaleFactor = 5
HotspotFactor = 3 #Switch_size/2



crrp_delay_hs=[]
crrp_delay_nohs=[]
crrp_delay_overall=[]
d_crrp_delay_hs=[[]]
d_crrp_delay_nohs=[[]]
d_crrp_delay_overall=[[]]


clqf_delay_hs=[]
clqf_delay_nohs=[]
clqf_delay_overall=[]
d_clqf_delay_hs=[[]]
d_clqf_delay_nohs=[[]]
d_clqf_delay_overall=[[]]


ga_delay_hs=[]
ga_delay_nohs=[]
ga_delay_overall=[]
d_ga_delay_hs=[[]]
d_ga_delay_nohs=[[]]
d_ga_delay_overall=[[]]






hs_load_list=[]
for i in range(0,Switch_size):
    hs_load_list.append((float(i)+1.0)*100.0/float(Switch_size))
    
import matplotlib.pyplot
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10.5, 3.5)
matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    
# print basic iterations
for j in range(0,len(test_iter_list)):  
    for i in range(0,len(test_load_list)):
        
  
        #print('speed up, iteration, hot shot factor',i,j,k)
        #print(test_load_index)
        #print(test_iter_index)
        '''
        filename = make_filename2(Duration, Switch_size, TrafficLoad, test_iter_list[j], -1, -1, i, 1)
        DATA = FILE_TO_ARRAY(filename)    
        gaFrom,gaTo,gaIn,gaOut = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_in_hs = 0
        accum_out_hs = 0
        for ii in range(0,len(gaFrom)):k
            if gaFrom[ii]==gaTo[ii]:
                accum_in_hs += gaIn[ii]
                accum_out_hs += gaOut[ii]
            accum_in += gaIn[ii]
            accum_out += gaOut[ii]
     
        thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
        ga_thruput_hs.append(thruput_hs)
        thruput = float(accum_out)*100.0/float(accum_in)
        ga_thruput.append(thruput)
            

        filename = make_filename2(Duration, Switch_size, TrafficLoad, test_iter_list[j], 5500, 5, i, 1)
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
        
        
        
        
        
        filename = make_filename(Duration, Switch_size, TrafficLoad, test_iter_list[j], Delay_threshold, Que_threshold, i, 2)
        DATA = FILE_TO_ARRAY(filename)    
        lqfFrom,lqfTo,lqfIn,lqfOut = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_in_hs = 0
        accum_out_hs = 0
        for ii in range(0,len(lqfFrom)):
            if lqfFrom[ii]==lqfTo[ii]:
                accum_in_hs += lqfIn[ii]
                accum_out_hs += lqfOut[ii]
            accum_in += lqfIn[ii]
            accum_out += lqfOut[ii]
     
        thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
        lqf_thruput_hs.append(thruput_hs)
        thruput = float(accum_out)*100.0/float(accum_in)
        lqf_thruput.append(thruput)
            
        filename = make_filename(Duration, Switch_size, TrafficLoad, test_iter_list[j], Delay_threshold, Que_threshold, i, 3)
        DATA = FILE_TO_ARRAY(filename)    
        ldfFrom,ldfTo,ldfIn,ldfOut = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_in_hs = 0
        accum_out_hs = 0
        for ii in range(0,len(ldfFrom)):
            if ldfFrom[ii]==ldfTo[ii]:
                accum_in_hs += ldfIn[ii]
                accum_out_hs += ldfOut[ii]
            accum_in += ldfIn[ii]
            accum_out += ldfOut[ii]
     
        thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
        ldf_thruput_hs.append(thruput_hs)
        thruput = float(accum_out)*100.0/float(accum_in)
        ldf_thruput.append(thruput)
        '''     
        filename = make_filename2(Duration, Switch_size, test_load_list[i], test_iter_list[j], -1, -1, HotspotFactor, 1)
        DATA = FILE_TO_ARRAY(filename)    
        crrpFrom,crrpTo,crrpDelaySum, crrpCount = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_delay_hs = 0
        accum_count_hs = 0
        accum_delay_nohs = 0
        accum_count_nohs = 0
        accum_delay_overall = 0
        accum_count_overall = 0
        for ii in range(0,len(crrpFrom)):
            if crrpFrom[ii]==crrpTo[ii]:
                accum_delay_hs += crrpDelaySum[ii]
                accum_count_hs += crrpCount[ii]
            else :
                accum_delay_nohs += crrpDelaySum[ii]
                accum_count_nohs += crrpCount[ii]
            accum_delay_overall += crrpDelaySum[ii]
            accum_count_overall += crrpCount[ii]
     
        average_delay_hs = float(accum_delay_hs)/float(accum_count_hs)
        average_delay_nohs = float(accum_delay_nohs)/float(accum_count_nohs)
        average_delay_overall = float(accum_delay_overall)/float(accum_count_overall)
        ga_delay_hs.append(average_delay_hs)
        ga_delay_nohs.append(average_delay_nohs)
        ga_delay_overall.append(average_delay_overall)        
        
        
        

        filename = make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j], Delay_threshold, Que_threshold, HotspotFactor, 4, ScaleFactor)
        DATA = FILE_TO_ARRAY(filename)    
        crrpFrom,crrpTo,crrpDelaySum, crrpCount = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_delay_hs = 0
        accum_count_hs = 0
        accum_delay_nohs = 0
        accum_count_nohs = 0
        accum_delay_overall = 0
        accum_count_overall = 0
        for ii in range(0,len(crrpFrom)):
            if crrpFrom[ii]==crrpTo[ii]:
                accum_delay_hs += crrpDelaySum[ii]
                accum_count_hs += crrpCount[ii]
            else :
                accum_delay_nohs += crrpDelaySum[ii]
                accum_count_nohs += crrpCount[ii]
            accum_delay_overall += crrpDelaySum[ii]
            accum_count_overall += crrpCount[ii]
     
        average_delay_hs = float(accum_delay_hs)/float(accum_count_hs)
        average_delay_nohs = float(accum_delay_nohs)/float(accum_count_nohs)
        average_delay_overall = float(accum_delay_overall)/float(accum_count_overall)
        
        #print(accum_delay_hs, accum_count_hs, average_delay_hs)
        crrp_delay_hs.append(average_delay_hs)
        crrp_delay_nohs.append(average_delay_nohs)
        crrp_delay_overall.append(average_delay_overall)
   
        
        filename = make_filename3(Duration, Switch_size, test_load_list[i], test_iter_list[j], Delay_threshold, Que_threshold, HotspotFactor, 4)
        DATA = FILE_TO_ARRAY(filename)    
        clqfFrom,clqfTo,clqfDelaySum, clqfCount = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_delay_hs = 0
        accum_count_hs = 0
        accum_delay_nohs = 0
        accum_count_nohs = 0
        accum_delay_overall = 0
        accum_count_overall = 0
        for ii in range(0,len(clqfFrom)):
            if clqfFrom[ii]==clqfTo[ii]:
                accum_delay_hs += clqfDelaySum[ii]
                accum_count_hs += clqfCount[ii]
            else :
                accum_delay_nohs += clqfDelaySum[ii]
                accum_count_nohs += clqfCount[ii]
            accum_delay_overall += clqfDelaySum[ii]
            accum_count_overall += clqfCount[ii]
     
        average_delay_hs = float(accum_delay_hs)/float(accum_count_hs)
        average_delay_nohs = float(accum_delay_nohs)/float(accum_count_nohs)
        average_delay_overall = float(accum_delay_overall)/float(accum_count_overall)
        
        #print(accum_delay_hs, accum_count_hs, average_delay_hs)
        clqf_delay_hs.append(average_delay_hs)
        clqf_delay_nohs.append(average_delay_nohs)
        clqf_delay_overall.append(average_delay_overall)

     
        '''            
        filename = make_filename(Duration, Switch_size, TrafficLoad, test_iter_list[j], Delay_threshold, Que_threshold, i, 5)
        DATA = FILE_TO_ARRAY(filename)    
        cldfFrom,cldfTo,cldfIn,cldfOut = MAKE_number_ARRAY2(DATA)
        accum_in = 0
        accum_out = 0
        accum_in_hs = 0
        accum_out_hs = 0
        for ii in range(0,len(cldfFrom)):
            if cldfFrom[ii]==cldfTo[ii]:
                accum_in_hs += cldfIn[ii]
                accum_out_hs += cldfOut[ii]
            accum_in += cldfIn[ii]
            accum_out += cldfOut[ii]
     
        thruput_hs = float(accum_out_hs)*100.0/float(accum_in_hs)
        cldf_thruput_hs.append(thruput_hs)
        thruput = float(accum_out)*100.0/float(accum_in)
        cldf_thruput.append(thruput)
        '''
    if j==0:
        #d_ga_thruput_hs[j] = ga_thruput_hs
        #d_ga_thruput[j] = ga_thruput
        #d_gapa_thruput_hs[j] = gapa_thruput_hs
        #d_gapa_thruput[j] = gapa_thruput
        #d_lqf_thruput_hs[j] = lqf_thruput_hs
        #d_lqf_thruput[j] = lqf_thruput
        #d_ldf_thruput_hs[j] = ldf_thruput_hs
        #d_ldf_thruput[j] = ldf_thruput
        d_crrp_delay_hs[j] = crrp_delay_hs
        d_crrp_delay_nohs[j] = crrp_delay_nohs
        d_crrp_delay_overall[j] = crrp_delay_overall
        
        d_clqf_delay_hs[j] = clqf_delay_hs
        d_clqf_delay_nohs[j] = clqf_delay_nohs
        d_clqf_delay_overall[j] = clqf_delay_overall
        
        d_ga_delay_hs[j] = ga_delay_hs
        d_ga_delay_nohs[j] = ga_delay_nohs
        d_ga_delay_overall[j] = ga_delay_overall
        #d_cldf_thruput_hs[j] = cldf_thruput_hs
        #d_cldf_thruput[j] =cldf_thruput

    else:
        #d_ga_thruput_hs.append(ga_thruput_hs)
        #d_ga_thruput.append(ga_thruput)
        #d_gapa_thruput_hs.append(gapa_thruput_hs)
        #d_gapa_thruput.append(gapa_thruput)
        #d_lqf_thruput_hs.append(lqf_thruput_hs)
        #d_lqf_thruput.append(lqf_thruput)
        #d_ldf_thruput_hs.append(ldf_thruput_hs)
        #d_ldf_thruput.append(ldf_thruput)
        d_crrp_delay_hs.append(crrp_delay_hs)
        d_crrp_delay_nohs.append(crrp_delay_nohs)
        d_crrp_delay_overall.append(crrp_delay_overall)
                
        d_clqf_delay_hs.append(clqf_delay_hs)
        d_clqf_delay_nohs.append(clqf_delay_nohs)
        d_clqf_delay_overall.append(clqf_delay_overall)
                
        
        d_ga_delay_hs.append(ga_delay_hs)
        d_ga_delay_nohs.append(ga_delay_nohs)
        d_ga_delay_overall.append(ga_delay_overall)      

        #d_cldf_thruput_hs.append(cldf_thruput_hs)
        #d_cldf_thruput.append(cldf_thruput)



        
        


    #for i in range(0,5):'''
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='upper left')

    #plt.plot(hs_load_list,ga_thruput_hs,'bv--')
    #plt.plot(hs_load_list,gapa_thruput_hs)
    plt.semilogy(test_load_list,crrp_delay_hs,'ro--')
    plt.semilogy(test_load_list,crrp_delay_nohs,'bv--')
    plt.semilogy(test_load_list,crrp_delay_overall,'g^--')
    #plt.plot(hs_load_list,cldf_thruput_hs)
    #plt.title('Average Delay   - switch:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor) + 'iter:'+str(test_iter_list[i]), fontsize=14)
    plt.title('C-RRP Delay - SW:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor)+' iter:'+str(test_iter_list[j]), fontsize=14)
    plt.xlabel('load (%)', fontsize=14)
    plt.ylabel('average delay (TS)', fontsize=14,)
    plt.legend(('hotspot','non-hotspot','overall'))
    #plt.ylim(0,3)
    plt.show() 


    #for i in range(0,5):'''
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='upper left')

    #plt.plot(hs_load_list,ga_thruput_hs,'bv--')
    #plt.plot(hs_load_list,gapa_thruput_hs)
    plt.semilogy(test_load_list,clqf_delay_hs,'ro--')
    plt.semilogy(test_load_list,clqf_delay_nohs,'bv--')
    plt.semilogy(test_load_list,clqf_delay_overall,'g^--')
    #plt.plot(hs_load_list,cldf_thruput_hs)
    #plt.title('Average Delay   - switch:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor) + 'iter:'+str(test_iter_list[i]), fontsize=14)
    plt.title('C-LQF Delay - SW:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor)+' iter:'+str(test_iter_list[j]), fontsize=14)
    plt.xlabel('load (%)', fontsize=14)
    plt.ylabel('average delay (TS)', fontsize=14,)
    plt.legend(('hotspot','non-hotspot','overall'))
    #plt.ylim(0,3)
    plt.show() 
    



    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='upper left')

    #plt.plot(hs_load_list,ga_thruput_hs,'bv--')
    #plt.plot(hs_load_list,gapa_thruput_hs)
    plt.semilogy(test_load_list,ga_delay_hs,'ro--')
    plt.semilogy(test_load_list,ga_delay_nohs,'bv--')
    plt.semilogy(test_load_list,ga_delay_overall,'g^--')
    #plt.plot(hs_load_list,cldf_thruput_hs)
    #plt.title('Average Delay   - switch:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor) + 'iter:'+str(test_iter_list[i]), fontsize=14)
    plt.title('GA Delay - SW:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor)+' iter:'+str(test_iter_list[j]), fontsize=14)
    plt.xlabel('load (%)', fontsize=14)
    plt.ylabel('average delay (TS)', fontsize=14,)
    plt.legend(('hotspot','non-hotspot','overall'))
    #plt.ylim(0,3)
    plt.show() 
    '''
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')
    #plt.plot(hs_load_list,ga_thruput,'bv--')
    #plt.plot(hs_load_list,gapa_thruput)
    plt.plot(hs_load_list,crrp_thruput,'ro--')
    #plt.plot(hs_load_list,cldf_thruput)
    plt.title('Overall throughput - switch:'+str(Switch_size)+' scale:'+str(test_scale_list[j]), fontsize=14)
    plt.xlabel('Hot Spot Load (%)', fontsize=14)
    plt.ylabel('Throughput (%)', fontsize=14,)
    plt.ylim(80,105)
    plt.xlim(0,100)
    plt.legend(('C-RRP'))
    plt.show() 
    '''



    crrp_delay_hs=[]
    crrp_delay_nohs=[]
    crrp_delay_overall=[]
    
    clqf_delay_hs=[]
    clqf_delay_nohs=[]
    clqf_delay_overall=[]

    ga_delay_hs=[]
    ga_delay_nohs=[]
    ga_delay_overall=[]

 
for i in range(0,len(test_iter_list)):  
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5.5, 3.5)
    matplotlib.rc('legend', fontsize=14, labelspacing=0.2, handlelength=2, frameon=False, loc='upper left')

    #plt.plot(hs_load_list,ga_thruput_hs,'bv--')
    #plt.plot(hs_load_list,gapa_thruput_hs)
    plt.semilogy(test_load_list,d_ga_delay_overall[i][:],'g^--')
    plt.semilogy(test_load_list,d_clqf_delay_overall[i][:],'bv--')
    plt.semilogy(test_load_list,d_crrp_delay_overall[i][:],'ro--')
    #plt.plot(hs_load_list,cldf_thruput_hs)
    #plt.title('Average Delay   - switch:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor) + 'iter:'+str(test_iter_list[i]), fontsize=14)
    plt.title('Average Delay - SW:'+str(Switch_size)+' scale:'+str(ScaleFactor)+' HS:'+str(HotspotFactor)+' iter:'+str(test_iter_list[i]), fontsize=14)
    plt.xlabel('load (%)', fontsize=14)
    plt.ylabel('average delay (TS)', fontsize=14,)
    plt.legend(('GA','C-LQF','C-RRP'))
    #plt.ylim(0,3)
    plt.show() 

    







    
    