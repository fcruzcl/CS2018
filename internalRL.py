# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:43:35 2013

@author: cruz
version 0.1.3: this version was used to create the file in classic RL
scenario which is used to make the plot after. That plot was used in 
ICDL-EpiRob paper.

The plots are created using the script called makePlot2 (function plot1) 
where this approach is compared with reinforcement learning + affordances
scenario.

An future improvement would be to save every single produced data and thus
not using the averaged data anymore. Another option could be create different
files that means dont overwrite them, or develop a short script to analize
those data.
"""
#Libraries Declaration
from classes.Scenario import Scenario
from classes.Agent import Agent
from classes.DataFiles import DataFiles
from classes.Plot import Plot

#******************************************************************************

#main
if __name__ == "__main__":
    print('RL for cleaning a table is running .... ')
    iterations = 10000
    tries = 100
    filenameStepsRL = 'results/stepsRL2.csv'
    filenameRewardsRL = 'results/rewardsRL2.csv'
    filenameStepsRLAff = 'results/stepsRLAff.csv'
    filenameRewardsRLAff = 'results/rewardsRL.csv'

    #In autonomous RL are all 0 since neither interaction nor affordances are present
    affordances = 0 #1: RL + Affordances, 0: Autonomous RL
    teacherAgent = 0
    feedbackProbability = 0
    consistencyProbability = 0

    files = DataFiles()
    files.createFile(filenameStepsRL)
    files.createFile(filenameRewardsRL)
    scenario = Scenario()

    for i in range(tries):
        agent = Agent(scenario)
        [steps, rewards] = agent.train(iterations, affordances, teacherAgent, feedbackProbability, consistencyProbability)

        files.addToFile(filenameStepsRL, steps)
        files.addFloatToFile(filenameRewardsRL, rewards)
    #endfor
       
    plot = Plot()
    plot.plotRLvsRLA(filenameStepsRL, filenameStepsRLAff)
    print ("The end")
    
# end of main method
