# -*- coding:utf-8 -*-
import sys 
sys.path.append('../Feature')
from Tkinter import *
import tkFileDialog
import SEAM
import SAM
import SYNAN
import DISAM
import SYNERR
import pickle
import numpy as np
from Core import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.window_init()
        self.createWidgets()
        self.pack()

    def window_init(self):
        self.master.title('Automated Essay Scoring System')
        width,height = 950,550
        self.master.geometry("{}x{}".format(width, height)+"+150+100")
        self.master.resizable(width=False, height=False)
    
    def createWidgets(self):
        #system frame
        self.systemframe = Frame(self)
        self.systemframe.pack(side=LEFT,expand=YES,fill='both')

        #essay frame
        self.essayframe = Frame(self.systemframe,bg='black')
        self.essayframe.pack(side= TOP,expand=YES,fill='both',pady=5)
        self.scroll = Scrollbar(self.essayframe)
        self.text = Text(self.essayframe)
        self.scroll.pack(side=RIGHT, fill='y')
        self.text.pack(side=LEFT, fill='y')
        self.scroll.config(command=self.text.yview)
        self.text.config(yscrollcommand=self.scroll.set)
        str = '''Dear, @ORGANIZATION1 concerned with an issue that people are using computers and not exersising. This is a very bad thing and we need to let the word out of what can happen if we don't exercise. The first reason why I disagree of the fact that computers help people is because computers really affect your health you will become fat you start having problems with your heart and it can lead to something fatal. Another reason why I think computers are healthy is because comptuers affect the economy alot in so many ways for example computers use up so much electricity it affects the power in all states. Computers also ruin jobs and this is very bad for the economy people will be addictied to computers for so long they wont bother looking for jobs they will just stay on the computer and gamble or buy stuff. The last reason why I think computers are bad is because of some of the illegal stuff people do on computers there are alot of cyber predetors that prey on kids and they end up finding the kid and kidnapp them. There are also other illegal stuff like bad websites or popups that can get you arrested so if kids go on computers they will not no what something means and they will click on it and it can traumitize a kid their whole life. So @NUM1 I hope you can take this letter into recognation and do something about it cause thes computers can affect my health your health and even your childrens health. That is why I have written this letter to you. If you agree with me I thank you.'''
        self.text.insert(INSERT, str)

        #advise frame
        self.adviseframe = Frame(self.systemframe, bg='black')
        self.adviseframe.pack(side=TOP,expand=YES,fill='x')
        self.text2 = Text(self.adviseframe)
        self.text2.pack(side=TOP,fill='both')
        self.text2.insert(INSERT, 'Stat  Info:\n\nSyxtax Error:\n\nSpell  Error:\n')
        self.text2.config(state=DISABLED)

        #user frame
        self.userframe = Frame(self,bg='gray')
        self.userframe.pack(side= RIGHT,expand=YES,fill='y',padx=5)
        #result frame
        self.resultframe = Frame(self.userframe)
        self.resultframe.pack(side= TOP,expand=YES,fill='y',padx=5)
        self.scoreframe = Frame(self.resultframe)
        self.scoreframe.pack(side= LEFT,expand=YES,fill='y',padx=5)
        self.canvasframe = Frame(self.resultframe)
        self.canvasframe.pack(side= RIGHT,expand=YES,fill='y')

        self.score1 = StringVar()
        self.score1.set("Outline idea  :0.0")
        self.score2 = StringVar()
        self.score2.set("Statistical   :0.0")
        self.score3 = StringVar()
        self.score3.set("Word use      :0.0")
        self.score4 = StringVar()
        self.score4.set("Key idea      :0.0")
        self.score5 = StringVar()
        self.score5.set("Syntax & Spell:0.0")
        self.totalscore = StringVar()
        self.totalscore.set("Total         :0.0")

        self.bgcolor = 'cornflowerblue'
        self.fontcolor = 'mediumvioletred'
        self.score1lable = Label(self.scoreframe, textvariable=self.score1, bg=self.bgcolor, fg=self.fontcolor, font=("Comic Sans MS", 15), justify="left")
        self.score1lable.pack(expand=YES,fill='both')
        self.score2lable = Label(self.scoreframe, textvariable=self.score2, bg=self.bgcolor, fg=self.fontcolor, font=("Comic Sans MS", 15), justify="left")
        self.score2lable.pack(expand=YES,fill='both')
        self.score3lable = Label(self.scoreframe, textvariable=self.score3, bg=self.bgcolor, fg=self.fontcolor, font=("Comic Sans MS", 15), justify="left")
        self.score3lable.pack(expand=YES,fill='both')
        self.score4lable = Label(self.scoreframe, textvariable=self.score4, bg=self.bgcolor, fg=self.fontcolor, font=("Comic Sans MS", 15), justify="left")
        self.score4lable.pack(expand=YES,fill='both')
        self.score5lable = Label(self.scoreframe, textvariable=self.score5, bg=self.bgcolor, fg=self.fontcolor, font=("Comic Sans MS", 15), justify="left")
        self.score5lable.pack(expand=YES,fill='both')
        self.totallable = Label(self.scoreframe, textvariable=self.totalscore, bg=self.bgcolor, fg=self.fontcolor, font=("Comic Sans MS", 15), justify="left")
        self.totallable.pack(expand=YES,fill='both')

        self.canvas = Canvas(self.canvasframe, width=200, height=480)
        self.canvas.pack()

        #button frame
        self.buttonframe = Frame(self.userframe)
        self.buttonframe.pack(side= TOP,expand=YES,fill='y')
        importbutton = Button(self.buttonframe, text="Import", command=self.get_essayfile)
        importbutton.pack(side= LEFT,expand=YES)
        utrasebutton = Button(self.buttonframe, text="Submit", command=self.get_scores)
        utrasebutton.pack(side= LEFT,expand=YES)
        nnbutton = Button(self.buttonframe, text="  NN  ", command=self.nn_score)
        nnbutton.pack(side= LEFT,expand=YES)
        lstmbutton = Button(self.buttonframe, text=" LSTM ", command=self.lstm_score)
        lstmbutton.pack(side= LEFT,expand=YES)
        cnnbutton = Button(self.buttonframe, text="  CNN ", command=self.cnn_score)
        cnnbutton.pack(side= LEFT,expand=YES)

    def get_essayfile(self):
        filename = tkFileDialog.askopenfilename()
        with open(filename, 'rb') as f:
            essay_file = f.read()
        self.text.delete('1.0', 'end')
        self.text.insert(INSERT, essay_file)

    def get_scores(self):
        essay = self.text.get('1.0', 'end')
        DataFileName = 'perfectessay.txt'
        seam_score = SEAM.performLSA(essay, DataFileName, ifesstxt=True)
        sam_score = SAM.performSA(essay, DataFileName, ifesstxt=True)
        synan_score = SYNAN.scoreSYN(essay, DataFileName, ifesstxt=True)
        disam_score = DISAM.scoreDiscourse(essay, DataFileName, ifesstxt=True)
        synerr_score = SYNERR.scoreSYNERR(essay, ifesstxt=True)
        calibrator = pickle.load(open("../Feature/calibrated_model.sav", 'rb'))
        scoref = int(calibrator.predict(np.array([[seam_score, sam_score, synan_score, disam_score, synerr_score]])))
        self.score1.set("Outline idea  :"+str(round(seam_score, 1)))
        self.score2.set("Statistical   :"+str(round(sam_score, 1)))
        self.score3.set("Word use      :"+str(round(synan_score, 1)))
        self.score4.set("Key idea      :"+str(round(disam_score, 1)))
        self.score5.set("Syntax & Spell:"+str(round(synerr_score, 1)))
        self.totalscore.set("Total         :"+str(round(scoref, 1)))

        length = 200
        x = 10
        y = 20
        height = 25
        padding = 62
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(x,y,x+length*(seam_score/12.0),y+height, fill="springgreen")
        self.canvas.create_rectangle(x,y+height+padding,x+length*(sam_score/12.0),y+height*2+padding, fill="violet")
        self.canvas.create_rectangle(x,y+2*(height+padding),x+length*(synan_score/12.0),y+2*(height+padding)+height, fill="royalblue")
        self.canvas.create_rectangle(x,y+3*(height+padding),x+length*(disam_score/12.0),y+3*(height+padding)+height, fill="deeppink")
        self.canvas.create_rectangle(x,y+4*(height+padding),x+length*(synerr_score/12.0),y+4*(height+padding)+height, fill="darkorange")
        self.canvas.create_rectangle(x,y+5*(height+padding),x+length*(scoref/12.0),y+5*(height+padding)+height, fill="gold")

        with open('./Database/synerr.txt', 'r') as f:
            synerr = f.read()
        with open('./Database/spellerr.txt', 'r') as f:
            spellerr = f.read()
        with open('./Database/stat.txt', 'r') as f:
            stat = f.read()
        self.text2.config(state=NORMAL)
        self.text2.delete('1.0', 'end')
        self.text2.insert(INSERT, stat+'\n')
        self.text2.insert(INSERT, synerr+'\n')
        self.text2.insert(INSERT, spellerr)
        self.text2.config(state=DISABLED)
    
    def nn_score(self):
        essay = self.text.get('1.0', 'end')
        with open('./Database/testessay.txt', 'w') as f:
            f.write(essay)
        _LOGFILENAME, timestamp = start_deepscore_core()
        essay_vector = Doc2Vec.inferEssayVector(_LOGFILENAME, './Database/testessay.txt')
        model = loadDeepScoreModel(_LOGFILENAME, "1_NN")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,300))))
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)

        self.totalscore.set("Total         :"+str(round(predicted_score, 1)))
        length = 200
        x = 10
        y = 20
        height = 25
        padding = 62
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(x,y+5*(height+padding),x+length*(predicted_score/12.0),y+5*(height+padding)+height, fill="gold")
    
    def lstm_score(self):
        essay = self.text.get('1.0', 'end')
        with open('./Database/testessay.txt', 'w') as f:
            f.write(essay)
        _LOGFILENAME, timestamp = start_deepscore_core()
        essay_vector = Doc2Vec.inferEssaySentenceVector(_LOGFILENAME, './Database/testessay.txt', 'Set1')
        model = loadDeepScoreModel(_LOGFILENAME, "1_LSTM")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,52,300))))
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)

        self.totalscore.set("Total         :"+str(round(predicted_score, 1)))
        length = 200
        x = 10
        y = 20
        height = 25
        padding = 62
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(x,y+5*(height+padding),x+length*(predicted_score/12.0),y+5*(height+padding)+height, fill="gold")

    def cnn_score(self):
        essay = self.text.get('1.0', 'end')
        with open('./Database/testessay.txt', 'w') as f:
            f.write(essay)
        _LOGFILENAME, timestamp = start_deepscore_core()
        essay_vector = Doc2Vec.inferEssaySentenceVector(_LOGFILENAME, './Database/testessay.txt', 'Set1')
        model = loadDeepScoreModel(_LOGFILENAME, "1_CNN")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,52,300,1))))
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)

        self.totalscore.set("Total         :"+str(round(predicted_score, 1)))
        length = 200
        x = 10
        y = 20
        height = 25
        padding = 62
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(x,y+5*(height+padding),x+length*(predicted_score/12.0),y+5*(height+padding)+height, fill="gold")

if __name__=='__main__':
    app = Application()
    app.mainloop()