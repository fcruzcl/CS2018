# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
version 0.1: this version was created from files used to prepare TAMD paper. 
The ideas comes from discussion after Osaka presentation.
Q-Values are saved to analyse the internal representation and try to make
explicit the implicit affordances during the reinforcement learning process.
Plots are created using heating maps in Excel with conditional formats.

"""
#Libraries Declaration
from classes.Scenario import Scenario
from classes.Agent import Agent
from classes.DataFiles import DataFiles
from classes.Plot import Plot

#******************************************************************************

#main
if __name__ == "__main__":
    print('IRL for cleaning a table is running .... ')
    episodes = 5000
    agents = 3
    filenameStepsRL = 'results/stepsRL.csv'
    filenameRewardsRL = 'results/rewardsRL445.csv'
    filenameStepsIRL = 'results/stepsIRL.csv'
    filenameRewardsIRL = 'results/rewardsIRL.csv'
    filenameVisitedStatesRL = 'results/visitedStatesRL.csv'
    filenameVisitedStatesIRL = 'results/visitedStatesIRL.csv'
    
    filenameTeacherAgent = 'results/QvaluesFinalRLAgent445.csv'

    files = DataFiles()
    #files.createFile(filenameStepsRL)
    #files.createFile(filenameRewardsRL)
    files.createFile(filenameStepsIRL)
    files.createFile(filenameRewardsIRL)
    #files.createFile(filenameVisitedStatesRL)
    files.createFile(filenameVisitedStatesIRL)

    scenario = Scenario()
    teacherAgent = Agent(scenario)
    teacherAgent.loadQValues(filenameTeacherAgent)

    """
    #Training with autonomous RL
    print('IRL is now training with autonomous RL')
    for i in range(agents):
        print('Training agent number: ' + str(i+1))
        teacherAgent = Agent(scenario)
        [steps, rewards, visitedStates] = teacherAgent.train(episodes)

        teacherAgent.saveQValues('results/QValues/QValuesInicialRLAgent'+str(i+1)+'.csv', teacherAgent.QInitial)
        teacherAgent.saveQValues('results/QValues/QvaluesFinalRLAgent'+str(i+1)+'.csv', teacherAgent.Q)

        files.addToFile(filenameStepsRL, steps)
        files.addFloatToFile(filenameRewardsRL, rewards)
        files.addToFile(filenameVisitedStatesRL, visitedStates)
    #endfor
    teacherAgent.saveQValues('results/QValuesInicialRL.csv', teacherAgent.QInitial)
    teacherAgent.saveQValues('results/QValuesFinalRL.csv', teacherAgent.Q)
    """

    #Training with interactive RL
    print('IRL is now training with interactive RL')
    feedback = 0.3
    for i in range(agents):
        print('Training agent number: ' + str(i+1))
        learnerAgent = Agent(scenario)
        [steps, rewards, visitedStates] = learnerAgent.train(episodes, teacherAgent, feedback)

        files.addToFile(filenameStepsIRL, steps)
        files.addFloatToFile(filenameRewardsIRL, rewards)
        files.addToFile(filenameVisitedStatesIRL, visitedStates)
    #endfor
    learnerAgent.saveQValues('results/QValuesInicialIRL.csv', learnerAgent.QInitial)
    learnerAgent.saveQValues('results/QValuesFinalIRL.csv', learnerAgent.Q)
       
    plot = Plot()
    #plot.plotRewards(filenameStepsRL)
    plot.plotRewards(filenameRewardsRL, filenameRewardsIRL)
    
    
    print ("The end")
    
# end of main method
