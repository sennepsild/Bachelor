from graphics import *
import random
import time
import numpy as np
import math


class Pong:
    done = False
    score = [0,0]
    posPuck1 = [50, 200]
    posPuck2 = [450, 200]
    posBall = [250,200]
    dirBall = [-1,0.1]

    puckPos = 1




    def getInit(self,render):
        self.done = False

        self.posPuck1 = [50, 200]
        self.posPuck2 = [450, 200]
        if(render):
            self.puck1.undraw()
            self.puck2.undraw()


            puck1 = Rectangle(Point(self.posPuck1[0] - 7, self.posPuck1[1] - 66.5),
                          Point(self.posPuck1[0] + 7, self.posPuck1[1] + 66.5))
            puck1.setFill('white')
            puck1.draw(self.win)

            puck2 = Rectangle(Point(self.posPuck2[0] - 7, self.posPuck2[1] - 40),
                          Point(self.posPuck2[0] + 7, self.posPuck2[1] + 40))
            puck2.setFill('white')
            puck2.draw(self.win)
            self.puck1 = puck1
            self.puck2 = puck2

        self.score = [0,0]
        self.posBall = [250,200]
        #self.dirBall = [-1 + (random.random() * 2), -1 + (random.random() * 2)]
        self.dirBall =[-1,0.1]
        if(render):
            self.ball.undraw()
            self.ball = Rectangle(Point(self.posBall[0] - 10, self.posBall[1] - 10),
                              Point(self.posBall[0] + 10, self.posBall[1] + 10))
            self.ball.setFill("white")
            self.ball.draw(self.win)
            self.scoreBoard.setText('%d   %d' % (self.score[0], self.score[1]))


        return np.array([self.puckPos,self.posBall[0],self.posBall[1],self.dirBall[0],self.dirBall[1]])

    def __init__(self):
        win = GraphWin('Pong', 500, 400, autoflush=False)
        win.setBackground("black")

        self.win = win

        ball = Rectangle(Point( self.posBall[0] - 10, self.posBall[1] - 10),
                         Point(self.posBall[0] + 10, self.posBall[1] + 10))
        ball.setFill("white")
        ball.draw(win)
        self.ball = ball

        scoreBoard = Text(Point(250, 30), '%d   %d' % (self.score[0], self.score[1]))
        scoreBoard.setTextColor("white")
        scoreBoard.setSize(20)
        scoreBoard.draw(self.win)

        self.scoreBoard = scoreBoard

        puck1 = Rectangle(Point(self.posPuck1[0] - 7, self.posPuck1[1] - 66.5),
                          Point(self.posPuck1[0] + 7, self.posPuck1[1] + 66.5))
        puck1.setFill('white')
        puck1.draw(win)

        puck2 = Rectangle(Point(self.posPuck2[0] - 7, self.posPuck2[1] - 40),
                          Point(self.posPuck2[0] + 7, self.posPuck2[1] + 40))
        puck2.setFill('white')
        puck2.draw(win)
        self.puck1 = puck1
        self.puck2 = puck2


        self.win.update()

    def mag(self,x):
        return math.sqrt(sum(i ** 2 for i in x))

    def normalize(self,x):
        return [x[0]/self.mag(x),x[1]/self.mag(x)]

    def MovePuck1(self,dir,render):
        if self.puckPos == 2 and dir >0:
            dir = 0

        elif self.puckPos == 0 and dir <0:
            dir = 0
        else:
            self.puckPos += dir

        dir *= 133



        if (render) :
            self.puck1.move(0,dir)
        self.posPuck1[1] += dir

    def MovePuck2(self, dir, render):
        if self.posPuck2[1]< 0 and dir <0:
            dir =0
            #self.done = True
        if self.posPuck2[1]>400 and dir >0:
            dir =0
            #self.done = True

        if (render):
            self.puck2.move(0, dir)
        self.posPuck2[1] += dir









    def Game(self,render,action1,action2):

        #if( sum(self.dirBall) !=1):
           # self.dirBall = self.normalize(self.dirBall)
        reward = 0


        if(action1 ==2):
            action1 =-1




        self.MovePuck1(action1, render)
        if(self.posBall[1]+10> self.posPuck2[1] ):
            self.MovePuck2(1, render)
        elif(self.posBall[1]-10< self.posPuck2[1] ):
            self.MovePuck2(-1, render)
        else:
            self.MovePuck2(0, render)
        #check if out and score
        if(self.posBall[0] > 440 or self.posBall[0] < 60):

            if self.posBall[0] > 440:
                self.score[1] += 1
                #reward = 100
            else:
                self.score[0] += 1
                #reward = -100
            if self.score[0] >= 1 or self.score[1] >=1:
                self.done = True

            self.posBall = [250, 200]
            #self.dirBall = [-1 +(random.random()*2),-1 +(random.random()*2)]
            self.dirBall = [-1,0.1]
            if render:
                self.ball.undraw()
                self.ball = Rectangle(Point( self.posBall[0] - 10, self.posBall[1] - 10), Point(self.posBall[0] + 10, self.posBall[1] + 10))
                self.ball.setFill("white")
                self.ball.draw(self.win)
                self.scoreBoard.setText('%d   %d' % (self.score[0], self.score[1]))


        if self.posBall[1] >= 400-5 or self.posBall[1] <= 5:
            self.dirBall[1] = self.dirBall[1] * -1

        if   (self.posPuck1[0]-17 <=self.posBall[0] <= self.posPuck1[0]+17) and (self.posPuck1[1]- 76.5 <= self.posBall[1]<= self.posPuck1[1]+ 76.5) :
            self.dirBall[1] += (2 / ((self.posBall[1] - self.posPuck1[1])+0.01))*-1
            reward = 1
            #self.dirBall[1] = self.dirBall[1] * -1
            self.dirBall[0] = self.dirBall[0] * -1


        if   (self.posPuck2[0]-17 <=self.posBall[0] <= self.posPuck2[0]+17) and (self.posPuck2[1]- 50 <= self.posBall[1]<= self.posPuck2[1]+ 50) :
            self.dirBall[1] += (2 / ((self.posBall[1] - self.posPuck2[1])+0.01))*-1

            #self.dirBall[1] = self.dirBall[1] * -1
            self.dirBall[0] = self.dirBall[0] * -1





        self.posBall[0] += self.dirBall[0]
        self.posBall[1] += self.dirBall[1]



        if render :


            self.ball.move(self.dirBall[0], self.dirBall[1])
            self.win.update()
            key = self.win.checkKey()


            if key == "w":
                self.MovePuck1(-1,render)
            elif key == "s":
                self.MovePuck1(1, render)

            if key == "i":
                self.MovePuck2(-1,render)
            elif key == "k":

                self.MovePuck2(1, render)










        else:
            self.win.close()

        return np.array([self.puckPos,self.posBall[0],self.posBall[1],self.dirBall[0],self.dirBall[1]]),np.array( [reward]),self.done







#main

#env = Pong()
#while True:

    #env.Game(True,0,0)
    #print("running")
    #time.sleep(0.01)
