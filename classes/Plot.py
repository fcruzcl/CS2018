# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
"""

import matplotlib.pyplot as plt
import numpy as np
import Variables
from DataFiles import DataFiles


class Plot(object):
    
    def __init__(self):
        self.alpha = 0.3#0.1 #0.7
        self.gamma = 0.9 #0.4
        self.epsilon = 0.1 #0.25
        self.L = 0.5 # Cambiar al mejor para estudio
        self.C = 1
    #end of __init__ method

    def averageData(self, filename, isFloat=False):
        files = DataFiles()
        if isFloat:
            steps = files.readFloatFile(filename)
        else:
            steps = files.readFile(filename)
        #endif
        iterations = len(steps[0])
        tries = len(steps)
    
        avgSteps = np.zeros(iterations)
        success = np.zeros(iterations)
    
        for j in range(iterations):
            acum = 0.0
            cont = 0
            for i in range(tries):
                if steps[i][j] != 0:
                    acum += steps[i][j]
                    cont += 1
                #endif
            #endfor
            success[j] = cont
            if cont != 0:
                avgSteps[j] = acum / cont
            #endif
        #endfor
        return avgSteps, steps, success
    #end of method averageData

    def plotSteps(self, stepsRL, stepsIRL):
        [avgStepsRL, stepsRL, successRL] = self.averageData(stepsRL) 
        [avgStepsIRL, stepsIRL, successIRL] = self.averageData(stepsIRL)
        convolveSet = 30
        convolveAvgStepsRL = np.convolve(avgStepsRL, np.ones(convolveSet)/convolveSet)
        convolveAvgStepsIRL = np.convolve(avgStepsIRL, np.ones(convolveSet)/convolveSet)


        tam = 16 #Fontsize
        plt.rcParams['font.size'] = tam
        plt.rc('xtick', labelsize=12) 
        plt.rc('ytick', labelsize=12) 
        
        plt.figure('Number of actions')
        plt.suptitle('Number of actions')

        plt.plot(avgStepsIRL, label = 'Average actions IRL', linestyle = '--', color =  'r')
        plt.plot(avgStepsRL, label = 'Average actions RL', linestyle = '--', color = 'y' )

        plt.plot(convolveAvgStepsIRL, linestyle = '-', color =  '0.2')
        plt.plot(convolveAvgStepsRL, linestyle = '-', color = '0.2' )

        plt.legend(loc=4,prop={'size':tam-4})
        plt.xlabel('Episodes')
        plt.ylabel('Actions')
        plt.grid()

        my_axis = plt.gca()
        #my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
        my_axis.set_xlim(0, len(avgStepsRL)/6)
        
        plt.show()
            
    #end of plotSteps method



    def plotRewards(self, rewardsRL, rewardsIRL):
        [avgRewardsRL, rewardsRL, successRL] = self.averageData(rewardsRL, isFloat=True) 
        [avgRewardsIRL, rewardsIRL, successIRL] = self.averageData(rewardsIRL, isFloat=True)
        convolveSet = 30
        convolveAvgRewardsRL = np.convolve(avgRewardsRL, np.ones(convolveSet)/convolveSet)
        convolveAvgRewardsIRL = np.convolve(avgRewardsIRL, np.ones(convolveSet)/convolveSet)


        fig, ax = plt.subplots()

        tam = 16 #Fontsize
        plt.rcParams['font.size'] = tam
        plt.rc('xtick', labelsize=12) 
        plt.rc('ytick', labelsize=12) 
        
        

        ax.plot(avgRewardsIRL, label = 'Average reward IRL', linestyle = '--', color =  'r')
        ax.plot(avgRewardsRL, label = 'Average reward RL', linestyle = '--', color = 'y' )

        ax.plot(convolveAvgRewardsIRL, linestyle = '-', color =  '0.2')
        ax.plot(convolveAvgRewardsRL, linestyle = '-', color = '0.2' )
        #plt.plot(avgStepsRLAff, label = 'RL with affordances', marker = '.', linestyle = '-', color =  'b')

        ax.set_title('Collected reward')
        ax.legend(loc=4,prop={'size':tam-4})
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.grid()

        ax.set_ylim(Variables.punishment-0.8, Variables.reward)
        ax.set_xlim(convolveSet, len(avgRewardsRL)/6)
        #my_axis = ax.gca()
        #ax.set_ylim(Variables.punishment-0.8, Variables.reward)
        #ax.set_xlim(convolveSet, len(avgRewardsRL)/6)
        
        plt.show()
            
    #end of plotRewards method

    def plotExperienceOne(self, fileNameRL, agentNumber = 0):
        dataRL = np.genfromtxt(fileNameRL, delimiter=',')
        #states = len(dataRL[0])
        states = 53

        ind = np.arange(states)  # the x locations for the groups
        width = 0.75       # the width of the bars

        fig, ax = plt.subplots()

        meansRL = np.mean(dataRL, axis=0)
        stdRL = np.std(dataRL, axis=0)

        if agentNumber != 0:
            meansRL = dataRL[agentNumber-1]
            stdRL = None

        #ax.bar(ind, meansRL, width, label='RAlbert Einstein,Sherlock Holmes,Frankensteineinforcement Learning', color='y', yerr=stdRL)
        #ax.bar(ind, dataRL, width, label='Reinforcement Learning', color='y')
        ax.bar(ind, meansRL, width, color='y', yerr=stdRL)


        # add some text for labels, title and axes ticks
        ax.set_xlabel('States', fontsize=20)
        ax.set_ylabel('Number of times', fontsize=20)
        ax.set_title('Experienced States', fontsize=24)
        ax.set_xticks(ind + width/2)
        ax.set_xticklabels([str(y) for y in range(1,states+1)])
        #ax.set_ylim(0, maxData*1.01)
        #ax.set_ylim(0, 3000)
        ax.set_xlim(-width/4, states)

        #ax.legend(loc=1,prop={'size':18})
        
        #major_ticks = np.arange(0, maxData, 500)
        #minor_ticks = np.arange(0, maxData, 50)
        #ax.set_yticks(major_ticks)                                                       
        #ax.set_yticks(minor_ticks, minor=True)                                           
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)    

        
        for rect,val in zip(ind, meansRL):
        #for rect,val in zip(ind, dataRL):
            ax.text(rect+0.4, val*1.005, '%d' % val, ha='center', va='bottom')

        #for rect,val in zip(ind, meansIRL):
            #ax.text(rect+0.7, val*1.005, '%d' % val, ha='center', va='bottom')
        
        
        plt.show()


            
    def plotExperience(self, fileNameRL, filenameIRL, teacherNumber = 0, learnerNumber = 0):
        dataRL = np.genfromtxt(fileNameRL, delimiter=',')
        dataIRL = np.genfromtxt(filenameIRL, delimiter=',')
        #states = len(dataRL[0])
        states = 53

        ind = np.arange(states)  # the x locations for the groups
        width = 0.45       # the width of the bars

        fig, ax = plt.subplots()

        meansRL = np.mean(dataRL, axis=0)
        stdRL = np.std(dataRL, axis=0)

        if teacherNumber != 0:
            meansRL = dataRL[teacherNumber-1]
            stdRL = None

        #ax.bar(ind, meansRL, width, label='Reinforcement Learning', color='y', yerr=stdRL)
        #ax.bar(ind, meansRL, width, label='Reinforcement Learning', color='#F5F6CE', yerr=stdRL)
        #ax.bar(ind, meansRL, width, label='Reinforcement Learning', color='#D8D8D8', yerr=stdRL)
        ax.bar(ind, meansRL, width, label='Specialist agent experience', color='#D8D8D8', yerr=stdRL) #blue: #F5F6CE

        meansIRL = np.mean(dataIRL, axis=0)
        stdIRL = np.std(dataIRL, axis=0)

        if learnerNumber != 0:
            meansIRL = dataIRL[learnerNumber-1]
            stdIRL = None
        
        #linestyle = {"linestyle":"--", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":10}
        #ax.bar(ind+width, meansIRL, width, label='Interactive Reinforcement Learning', color='#FA8258', yerr=stdIRL, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2))
        ax.bar(ind+width, meansIRL, width, label='Polymath agent experience', color='#5198DF', yerr=stdIRL, ecolor='b')

        maxData = max(np.max(meansRL), np.max(meansIRL))
        maxData = max(np.max(dataRL), np.max(meansIRL))
        # add some text for labels, title and axes ticks
        ax.set_xlabel('States', fontsize=20)
        ax.set_ylabel('Frequency', fontsize=20)
        ax.set_title('Experienced States', fontsize=24)
        ax.set_xticks(ind + width)
        ax.set_xticklabels([str(y) for y in range(1,states+1)])
        #ax.set_ylim(0, maxData*1.01)
        ax.set_ylim(0, 5000)
        ax.set_xlim(-width/4, states)

        ax.legend(loc=1,prop={'size':18})
        
        major_ticks = np.arange(0, maxData, 500)
        minor_ticks = np.arange(0, maxData, 50)
        ax.set_yticks(major_ticks)                                                       
        ax.set_yticks(minor_ticks, minor=True)                                           
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)    

#        i=0
#        for rect,val in zip(ind, meansRL):
#        #for rect,val in zip(ind, dataRL):
#            if i%2 == 0:
#                ax.text(rect+0.2, val*1.005, '%d' % val, ha='center', va='bottom')
#            i=i+1
        """
        i = 0        
        for rect,val in zip(ind, meansIRL):
            #if i%2 == 1:
            ax.text(rect+0.7, val*1.005, '%d' % val, ha='center', va='bottom')
            i=i+1
        """        
        
        plt.show()
    
    def plotAllRewards(self, filename, labelTxt, colorTxt, labels=False, title=None):
        [avgData, allData, successData] = self.averageData(filename, True) 
        convolveSet = 30
        convolveData = np.convolve(avgData, np.ones(convolveSet)/convolveSet)

        if labels:
            tam = 18 #Fontsize
            plt.rcParams['font.size'] = tam
            plt.rc('xtick', labelsize=16) 
            plt.rc('ytick', labelsize=16) 
            plt.figure('RewardsF1.0C' + title)
            #plt.suptitle('Collected rewards, feedback = 0.25, consistency = ' + title)
            plt.suptitle('feedback = 1.0, consistency = ' + title, fontsize=22)
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.grid()
            my_axis = plt.gca()
            #my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
            my_axis.set_xlim(convolveSet, len(avgData)/6)

        #plt.plot(avgData, label = labelTxt, linestyle = '--', color =  colorTxt)
        plt.plot(convolveData, label = labelTxt, linewidth=1, linestyle = '-', color =  colorTxt)
        #plt.legend(loc=2,prop={'size':18}) #2:top-left, 4: bottom-right

        
        plt.show()
        
            
#end of class Plot


if __name__ == "__main__":
    plot = Plot()
    #results = '../results/presentationAdvisorLearnerDecisionsFeedback/Feedback0.0/'
    results = '../results/July6/Feedback1.0/'
    #plot.plotSteps(results + 'stepsRL.csv', results + 'stepsIRL.csv')
    
    
    consistency = '1.0'
    plot.plotAllRewards(results + 'rewardsRL.csv', 'RL', 'y', True, consistency)
    color = 'kgbcr'
    colorCount = 0
    for i in frange(0.0,1.0,0.25):
        plot.plotAllRewards(results + 'C' + consistency + 'F' + str(i) + 'rewardsIRL.csv', 'obedience = ' + str(i), color[colorCount])
        colorCount += 1
        
    #plot.plotRewards(results + 'rewardsRL.csv', results + 'C1F1rewardsIRL.csv')
    
    #for i in frange(0.0,1.0,0.25):
    #    plot.plotRewards(results + 'rewardsRL.csv', results + 'C1.0F' + str(i) + 'rewardsIRL.csv')

    #plot.plotExperience(results + 'visitedStatesRL.csv', results + 'C1F1visitedStatesIRL.csv')#, 57, 12)
    ###plot.plotExperience(results + 'June3.1/visitedStatesRL.csv', results + 'June16.2/C1F1visitedStatesIRL.csv', 57)#, 12)

    #plot.plotExperience(results + 'June3.1/visitedStatesRL.csv', results + 'June3.1/visitedStatesRL.csv', 13, 57)

    #plot.plotExperience(results + 'visitedStatesRL.csv', results + 'visitedStatesRL.csv', 13, 57)
    #plot.plotExperienceOne(results + 'visitedStatesRL.csv', 46)
