# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
version 0.1: this version was created from files used to prepare TAMD paper. 
The ideas comes from discussion after Osaka presentation.
Q-Values are saved to analyse the internal representation and try to make
explicit the implicit affordances during the reinforcement learning process.
Plots are created using heating maps in Excel with conditional formats.

I added some modifications to train many teachers and I then selected one.
Also changes in the reward to test the internal Q-values.
Also probability of select an advised action instead of selecting it.

version 0.2
In this version is intented to provide data about feedback to use it to train
a neural network in order to anticape the advice and determine until when
the advice is needed.

version 0.3
NEW Scenario proposed, more general and possible to extend it.
Modification in the main code, now is a class to run experiment with moduls
"""
#Libraries Declaration
from classes.Scenario import Scenario
from classes.Agent import Agent
from classes.DataFiles import DataFiles
from classes.Plot import Plot
import numpy as np

#******************************************************************************

class Experiment(object):
    def __init__(self, results, agents, episodes):
        self.files = DataFiles()
        self.resultsFolder = results
        self.scenario = Scenario()
        self.episodes = episodes
        self.agents = agents


    def trainTeachers(self):
        filenameStepsRL = self.resultsFolder + 'stepsRL.csv'
        filenameRewardsRL = self.resultsFolder + 'rewardsRL.csv'
        filenameVisitedStatesRL = self.resultsFolder + 'visitedStatesRL.csv'
        self.files.createFile(filenameStepsRL)
        self.files.createFile(filenameRewardsRL)
        self.files.createFile(filenameVisitedStatesRL)
        
        #Training with autonomous RL
        print('IRL is now training the teacher with autonomous RL')
        for i in range(self.agents):
            print('Training teacher agent number: ' + str(i+1))
            teacherAgent = Agent(self.scenario)
            [steps, rewards, visitedStates] = teacherAgent.train(self.episodes)
    
            teacherAgent.saveQValues(self.resultsFolder + 'QValues/QFinalRLAgent'+str(i+1)+'.csv', teacherAgent.Q)
    
            self.files.addToFile(filenameStepsRL, steps)
            self.files.addFloatToFile(filenameRewardsRL, rewards)
            self.files.addToFile(filenameVisitedStatesRL, visitedStates)
        #endfor
    #end trainTeacher method

    def trainLearners(self, feedback, consistency=1, follow=1):
        results = self.resultsFolder + 'C' + str(consistency) + 'F' + str(follow)
        filenameStepsIRL = results + 'stepsIRL.csv'
        filenameRewardsIRL = results + 'rewardsIRL.csv'
        filenameVisitedStatesIRL = results + 'visitedStatesIRL.csv'
        self.files.createFile(filenameStepsIRL)
        self.files.createFile(filenameRewardsIRL)
        self.files.createFile(filenameVisitedStatesIRL)
        
        #Loading teacher        
        bestTeacher = self.selectBestAgent()
        print 'The teacher selected is agent number: ' + str(bestTeacher)
        filenameTeacherAgent = self.resultsFolder + 'QValues/QFinalRLAgent' + str(bestTeacher) + '.csv'
        teacherAgent = self.loadTeacher(filenameTeacherAgent)

        #Training with interactive RL
        print('IRL is now training the learner with interactive RL')
        for i in range(self.agents):
            print('Training learner agent number: ' + str(i+1))
            learnerAgent = Agent(self.scenario)
            [steps, rewards, visitedStates] = learnerAgent.train(self.episodes, teacherAgent, feedback, consistency, follow)

            learnerAgent.saveQValues(self.resultsFolder + 'QValues/C' + str(consistency) + 'F' + str(follow) + 'QFinalIRLAgent'+str(i+1)+'.csv', learnerAgent.Q)
    
            self.files.addToFile(filenameStepsIRL, steps)
            self.files.addFloatToFile(filenameRewardsIRL, rewards)
            self.files.addToFile(filenameVisitedStatesIRL, visitedStates)

        #endfor
    #end trainLearner method
            
    def selectBestAgent(self):
        filename = self.resultsFolder + 'visitedStatesRL.csv'
        dataRL = np.genfromtxt(filename, delimiter=',')
        mean = np.mean(dataRL, axis=1)
        std = np.std(dataRL, axis=1)
        maxDifference = np.argmax(mean-std)
        teacher = maxDifference + 1
        meanVisit = mean[maxDifference]
        stdVisit = std[maxDifference]
        diffVisit = meanVisit - stdVisit
        return teacher, meanVisit, stdVisit, diffVisit
    #end selectBestAgent method

    def selectWorstAgent(self):
        filename = self.resultsFolder + 'visitedStatesRL.csv'
        dataRL = np.genfromtxt(filename, delimiter=',')
        mean = np.mean(dataRL, axis=1)
        std = np.std(dataRL, axis=1)
        minDifference = np.argmin(mean-std)
        teacher = minDifference + 1
        meanVisit = mean[minDifference]
        stdVisit = std[minDifference]
        diffVisit = meanVisit - stdVisit
        return teacher, meanVisit, stdVisit, diffVisit
    #end selectBestAgent method

    def loadTeacher(self, filename):
        teacherAgent = Agent(self.scenario)
        teacherAgent.loadQValues(filename)
        return teacherAgent
    #end loadTeacher method
        
#    def trainOnlyLearners(self, feedback, consistency=1, follow=1):
#        visitedStates = self.resultsFolder + 'visitedStatesRL.csv'
#        bestTeacher = self.selectBestTeacher(visitedStates)
#        print 'The teacher selected is agent number: ' + str(bestTeacher)
#    
#        filenameTeacherAgent = self.resultsFolder + 'QValues/QFinalRLAgent' + str(bestTeacher) + '.csv'
#        teacherAgent = self.loadTeacher(filenameTeacherAgent)
#               
#        self.trainLearner(teacherAgent, feedback, consistency, follow)
#        #bestLearner = self.selectBestLearner(self.resultsFolder + 'visitedStatesIRL.csv')
#        #print 'The learner selected is agent number: ' + str(bestLearner)
#           
#        #plot = Plot()
#        #plot.plotRewards(self.resultsFolder + 'rewardsRL.csv', self.resultsFolder + 'rewardsIRL.csv')
#        #plot.plotExperience(self.resultsFolder + 'visitedStatesRL.csv', self.resultsFolder + 'visitedStatesIRL.csv', bestTeacher, bestLearner)
#    #end trainOnlyLearners method
        

#main
if __name__ == "__main__":
    print('IRL for cleaning a table is running .... ')
    feedback = 0.25
    results = 'results/June6.1/'
    agents = 100
    episodes = 3000
    experiment = Experiment(results, agents, episodes)
    experiment.trainTeachers()
    print experiment.selectBestAgent()
    print experiment.selectWorstAgent()
    #experiment.trainLearners(feedback)

    
#    for j in frange(0.75,1.0,0.05):
#        for i in frange(0.0,1.0,0.25):
#            print 'Training with consistency probability of: '+ str(j)
#            print 'Training with following probability of: '+ str(i)
#            experiment.trainOnlyLearners(feedback, j,i)
    
    
    print ("The end")
    
# end of main method
