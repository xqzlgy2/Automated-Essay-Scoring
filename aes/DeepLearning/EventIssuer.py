from logwriter import *
import sys

class bcolors:
    PINK = '\033[95m'
    LIME = '\033[0;32m'
    YELLOW = '\033[93m'
    VIOLET = '\033[0;35m'
    BROWN = '\033[0;33m'
    INDIGO = "\033[0;34m"
    BLUE = "\033[0;34m"
    LIGHTPURPLE = '\033[1;35m'
    LIGHTRED = '\033[1;31m'
    NORMAL = '\033[0;37m'
    SHARP = '\033[1;30m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    SLEEP = '\033[90m'
    UNDERLINE = '\033[4m'


    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.END = ''


def issueMessage(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.NORMAL}{.BOLD}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors, bcolors))
    else:
        toprint = "{.NORMAL}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + text)


def issueSleep(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.SLEEP}{.BOLD}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors, bcolors))
    else:
        toprint = "{.SLEEP}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + text)

def issueSharpAlert(text, logfilename, highlight=False):
    if highlight:
        toprint = "{.BOLD}(Engine)" + " : " + text + "{.END}"
        print(toprint.format(bcolors, bcolors))
    else:
        toprint = "{.BOLD}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + text)


def issueError(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.FAIL}{.BOLD}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors, bcolors))
    else:
        toprint = "{.FAIL}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + text)

def issueWelcome(logfilename):
    with open(logfilename, 'a') as f:
        f.write(" __         __          \n|  \ _ _ _ (_  _ _  _ _ \n|__/(-(-|_)__)(_(_)| (- \n        |               ")
    print("\n\n")
    with open(logfilename, 'a') as f:
        f.write("\n\n(Engine) : Welcome to AES")
    toprint = "{.BLUE}{.BOLD}(Engine){.END}" + " : " + "Welcome to AES v0.1"
    print(toprint.format(bcolors, bcolors, bcolors))

def issueSuccess(text, logfilename, ifBold=False, highlight=False):
    if highlight:
        toprint = "{.LIME}{.BOLD}(Engine)" + " : " + text + "{.END}"
        print(toprint.format(bcolors, bcolors, bcolors))
    else:
        if ifBold:
            toprint = "{.LIME}{.BOLD}(Engine){.END}" + " : " + text
            print(toprint.format(bcolors, bcolors, bcolors))
        else:
            toprint = "{.LIME}(Engine){.END}" + " : " + text
            print(toprint.format(bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + text)

def genLogFile(logfilename, ts, strts):
    toprint = "{.LIME}{.BOLD}(Engine){.END}" + " : " + "Logging all events to " + str(ts)
    print(toprint.format(bcolors, bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + "Log File Created at : " + str(strts))
        f.write("\n(Engine) : " + "Logging all events to " + str(ts))


def issueWarning(text, logfilename, ifBold=False):
    if ifBold:
        toprint = "{.BROWN}{.BOLD}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors, bcolors))
    else:
        toprint = "{.BROWN}(Engine){.END}" + " : " + text
        print(toprint.format(bcolors, bcolors))
    with open(logfilename, 'a') as f:
        f.write("\n(Engine) : " + text)

def issueExit(logfilename, ts):
    toprint = "{.LIGHTPURPLE}{.BOLD}(Engine){.END}" + " : Shutting down the engine."
    print(toprint.format(bcolors, bcolors, bcolors))
    genpdfcmd = "python logwriter.py " + logfilename + " -S \"LOG FILE\" -A \"DeepScore Engine\" -o logs/DeepScore_Log_" + str(ts) + ".pdf"
    os.system(genpdfcmd)