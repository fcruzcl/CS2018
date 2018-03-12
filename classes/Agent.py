# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
"""

from DataFiles import DataFiles
import numpy as np
import Variables

class Agent(object):

    alpha = 0.3#0.1 #0.7
    gamma = 0.9 #0.4
    epsilon = 0.1 #0.25

    def __init__(self, scenario):
        self.scenario = scenario
        self.numberOfStates = self.scenario.getNumberOfStates()
        self.numberOfActions = self.scenario.getNumberOfActions()
        self.Q = np.random.uniform(0.0,0.01,(self.numberOfStates,self.numberOfActions))
        self.QInitial = np.copy(self.Q)
        self.visitedStates = np.zeros(self.numberOfStates)
        self.files = DataFiles()
#        self.filenameStates = 'datasets/states.csv'
#        self.filenameActions = 'datasets/actions.csv'
#        self.filenameNemonicos = 'datasets/nemonicos.csv'
#        self.files.createFile(self.filenameStates)
#        self.files.createFile(self.filenameActions)
#        self.files.createFile(self.filenameNemonicos)
        self.stateList = []
        self.actionList = []
        #self.actionDataset = [0] * self.numberOfActions
        self.feedbackOp = 0
        self.feedbackAmount = 0
        self.rightConfidence = 0
        self.badAdvice = 0
        #self.Q = np.zeros((self.numberOfStates,self.numberOfActions))
    #end of __init__ method

    def loadQValues(self, filename):
        self.Q = np.genfromtxt(filename, delimiter=',')
        
    def saveQValues(self, filename, QMatrix):
        self.files.createFile(filename)
        for i in range(self.numberOfStates):
            self.files.addFloatToFile(filename, QMatrix[i])

    def selectAction(self, state):
        if (np.random.rand() <= self.epsilon):
            action = np.random.randint(self.numberOfActions)
        else:
            action = np.argmax(self.Q[state,:])
        #endIf
        return action
    #end of selectAction method
        
    def actionByFeedback(self, state, teacherAgent, feedbackProbability, consistency, follow):
        
        if (np.random.rand() < feedbackProbability):

            """
            stateDataset = Variables.stateToFile[state]
            actionDataset = [0] * self.numberOfActions
            if stateDataset in self.stateList:
                pos = self.stateList.index(stateDataset)
                actionCoded = self.actionList[pos]
                action = actionCoded.index(1)
            else:
                #get advice
                actionAdvised = np.argmax(teacherAgent.Q[state,:])
                #ownActionrightConfidence = np.argmax(self.Q[state,:])
                action = actionAdvised
                #action = np.round(np.random.laplace(actionAdvised, 0.1)).astype(int)
                #if action < 0 or action >= self.numberOfActions:
                #    action = self.selectAction(state)
                #endIf
                #if action != actionAdvised:
                #    self.badAdvice += 1
                    #print self.feedbackAmount, state, actionAdvised, action, ownAction
                    
                self.feedbackAmount += 1

                self.stateList.append(stateDataset)
                
                actionDataset[actionAdvised] = 1
                self.actionList.append(actionDataset)
                
                #self.files.addToFile(self.filenameNemonicos, [state, actionAdvised, ownAction, action])
                #self.files.addToFile(self.filenameStates, Variables.stateToFile[state])
                #self.files.addToFile(self.filenameActions, actionToFile)
            """
            
            #get advice
            if (np.random.rand() < consistency):
                #good advice 
                advicedAction = np.argmax(teacherAgent.Q[state,:])
            else:
                #bad advice -- random advice
                advicedAction = np.random.randint(self.numberOfActions)
                
            self.feedbackOp += 1

            if (np.random.rand() < follow):
                self.feedbackAmount += 1
                action = advicedAction
            else:
                ownAction = np.argmax(self.Q[state,:])
                if ownAction == advicedAction:
                    self.rightConfidence += 1
                #endIf
                #if follow > 0:
                #    action = ownAction
                #else:
                action = self.selectAction(state)
            #endIf

        else:
            action = self.selectAction(state)
        #endIf
        return action
    #end of actionByFeedback
    
    def train(self, episodes, teacherAgent=None, feedbackProbability=0, consistency = 1, follow=1):
        contCatastrophic = 0
        contFinalReached = 0
        steps = np.zeros(episodes)
        rewards = np.zeros(episodes)
        
        for i in range(episodes):
            contSteps = 0
            accReward = 0
            self.scenario.resetScenario()
            state = self.scenario.getState()
            self.visitedStates[0] += 1
            action = self.actionByFeedback(state, teacherAgent, feedbackProbability, consistency, follow)

            #expisode
            while True:
                #perform action
                self.scenario.executeAction(action)
                contSteps += 1

                #get reward
                reward = self.scenario.getReward()
                accReward += reward
                #catastrophic state

                stateNew = self.scenario.getState()
                if stateNew != -1:
                    self.visitedStates[stateNew] += 1

                if reward == Variables.punishment:
                    contCatastrophic += 1
                    self.Q[state,action] = -0.1
                    break

                actionNew = self.actionByFeedback(stateNew, teacherAgent, feedbackProbability, consistency, follow)

                # updating Q-values
                self.Q[state, action] += self.alpha * (reward + self.gamma * 
                                         self.Q[stateNew,actionNew] - 
                                         self.Q[state,action])

                if reward == Variables.reward:
                    contFinalReached += 1
                    break


                state = stateNew
                action = actionNew
            #end of while
            steps[i] = contSteps
            rewards[i]=accReward
        #end of for
        #print self.feedbackOp, ',', self.feedbackAmount, ',', self.rightConfidence
        #stateArray = np.array(self.stateList)
        #actionArray = np.array(self.actionList)
        #print stateArray.shape
        #print actionArray.shape
        return steps,rewards, self.visitedStates
    #end of train method

#end of class Agent
