#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.90.3),
    on January 18, 2019, at 15:57
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import absolute_import, division
from psychopy import locale_setup, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding
from pathlib import Path  # for building paths
import collections  # to generate OrderedDict object for the TrialHandler

# Import hardware libraries
from psychopy import visual
from psychopy.hardware import joystick

# Import helper functions
from helper_functions import *

# experiment mode
# Should we show the debugging msgs on screen?
# In the present experiment, this means showing the credits on the screen
debugging = True
emulator_mode = 'Scan'

# Ensure that relative paths start from the same directory as this script
_thisDir = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(str(_thisDir))

# Store info about the experiment
if debugging:
    nRepeat = 3
    trial_duration = 30
    choice_duration = 20
else:
    nRepeat = 5
    trial_duration = 110
    choice_duration = 90

# get next subject number (lowest number not already in DATA directory)
data_path = _thisDir.parent.joinpath("DATA" + os.sep + "behavioral").resolve()
if os.path.isdir(str(data_path)) == False:
    os.mkdir(str(data_path))

subdirs = [o for o in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, o))]
subj_numbers = [int(s.split('-')[-1]) for s in subdirs]
subj_id = 0
while subj_id in subj_numbers:
    subj_id += 1

expName = 'habits_v_behavioral'  # from the Builder filename that created this script
expInfo = {'n_repeat_training': nRepeat,
           'credits_per_rnf': 40, 'cost_response': 1,
           'min_resp_bet_rnf': 5, 'max_resp_bet_rnf': 30, 'mean_resp_bet_rnf': 15, 'sigma_resp_bet_rnf': 5,
           'trial_duration': trial_duration, 'choice_duration': choice_duration,
           'schedule': 'RR', 'participant': subj_id, 'age': 18, 'gender': 'm', 'ethnicity': ''
           }

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# set data paths
part_path = data_path.joinpath('sub-{}'.format(expInfo['participant']))

# set condition balancing list
balancingList = 'ConditionsLists_habits_behavioral.xlsx'

# create participant subfolder if it does not already exist
if os.path.isdir(str(part_path)) == False:
    os.mkdir(str(part_path))

# create data file name + later add .psyexp, .csv, .log, etc.
filename = str(part_path) + os.sep + 'sub-{}_task-{}'.format(expInfo['participant'], expName)

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(
    name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

# fullscreen = not debugging
fullscreen = True
size = [1920, 1080] if not debugging else [800, 600]
# Setup the Window
win = visual.Window(
    size=size, fullscr=fullscreen, screen=1,
    allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0.506, 0.506, 0.506], colorSpace='rgb',
    blendMode='avg', useFBO=True)
wh_ratio = win.size[0] / win.size[1]  # get width-height ratio for scaling stimuli

# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

nBlocks = 2
nTrials = int(expInfo['n_repeat_training'])

# Create some handy timers
globalClock = core.Clock()  # to track the time since scan launched (is reset once between blocks)
routineClock = core.Clock()
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

# define durations of instructions, trials etc
fixed_durations = {
    "training": expInfo['trial_duration'],
    "consumption": 15,
    "choice_test": expInfo['choice_duration'],
}


# Counterbalance function
def CounterBalance(participant, conditionfile):
    """
    This function extracts the experimental conditions from the file 'ConditionsLists_habits2_v8.xlsx' which is provided with the task code.

    Inputs:
        - participant number (as a string)
        - conditionfile (xlsx file)

    Outputs:
        - OrderedDict object for data.trialList (TrialHandler)
    """
    parvalue = int(participant)  # get numeral value from text string
    start_idx = 5 * ((parvalue - 1) % 32)
    return data.importConditions(conditionfile, selection=str(start_idx) + ":" + str(start_idx + 5))


# end CounterBalance function

def show_debugging_stuff():
    ''' set text for stims to show every frame '''
    # shouldBeReinforced_txt.setText('Reinforce?: {:d}'.format(int(shouldBeReinforced))
    num_credits_txt.setText(str(credits))
    # Draw stim on screen
    # shouldBeReinforced_txt.draw()
    credits_title_txt.draw()
    num_credits_txt.draw()


def checkRnf(loop):
    ''' Once a response has been recorded -resp_made()-,
    this function checks whether it is reinforced, gives reward if shouldBeReinforced flag is True
    and records result for the csv '''
    global credits, shouldBeReinforced, nReinforcers, nResp_since_last_rnf, nResp_until_next_rnf  # globals because these are used outside

    if expInfo['schedule'] == 'RR':
        # shouldBeReinforced = np.random.binomial(1, 1 / float(expInfo['parameter']))
        if nResp_since_last_rnf >= nResp_until_next_rnf:
            shouldBeReinforced = 1
            loop.addData("nResp_untilRnf", nResp_until_next_rnf)

    if shouldBeReinforced:
        nResp_since_last_rnf = 0  # reset after reinforcer is delivered
        nResp_until_next_rnf = round(min(
            expInfo['max_resp_bet_rnf'],
            max(expInfo['min_resp_bet_rnf'],
                np.random.normal(expInfo['mean_resp_bet_rnf'], expInfo['sigma_resp_bet_rnf']))
        ))
        print("nResp until next rnf: " + str(nResp_until_next_rnf))
        nReinforcers += 1
        credits += float(expInfo['credits_per_rnf'])
        if phase == "training":  # light up coin if in training phase
            for frameN in range(20):  # show for bit more than half a sec (60 frames = 1 sec)
                triangle_resp.setOpacity(1)
                coin.setOpacity(1)
                win.flip()

    # # add data to csv file for training response
    loop.addData("wasRnf", shouldBeReinforced)
    shouldBeReinforced = 0


def resp_made(resp):
    global credits, nResp, nResp_this_trial, nResp1_this_trial, nResp2_this_trial, nResp_since_last_rnf  # credits var updates throughout the experiment
    ''' after a resp is made, record it for the csv file and other actions for current phase '''

    if phase == "training":
        # record resp and resp time for csv
        trialLoop.addData("event", "training key resp")
        trialLoop.addData("resp", resp)
        trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        trialLoop.addData("routineClock_t", str(t))
        trialLoop.addData("trial", nBlocks * nTrials * blockLoop.thisN + trialLoop.thisN)
        if resp in [corrResp]:
            # if valid response for stimulus
            nResp += 1
            nResp_since_last_rnf += 1
            credits -= float(expInfo['cost_response'])  # subtract cost of response from credits
            for nframes in range(5):  # show triangle
                triangle_resp.setOpacity(1)
                win.flip()
            checkRnf(trialLoop)  # check whether to reinforce
        thisExp.nextEntry()  # go to next entry

    elif phase == "consumption":
        if resp in [outcome_choice1, outcome_choice2]:
            # add to credits if response is valid and is valued
            if resp == outcome_choice1:
                if cond_val1 == 'val':
                    credits += float(expInfo['credits_per_rnf'])
            elif resp == outcome_choice2:
                if cond_val2 == 'val':
                    credits += float(expInfo['credits_per_rnf'])
            # record resp and resp time for csv
            blockLoop.addData("event", "consumption key resp")
            # blockLoop.addData("key_press", key_resp.keys)
            blockLoop.addData("resp", resp)
            blockLoop.addData("globalClock_t", str(globalClock.getTime()))
            blockLoop.addData("routineClock_t", str(t))
            thisExp.nextEntry()

    elif phase == "choice":
        if resp in [corrResp1, corrResp2]:
            if resp == corrResp1:
                nResp1_this_trial += 1
                credits -= float(expInfo['cost_response'])  # subtract response cost
                if cond_val1 == 'val':
                    nResp_since_last_rnf += 1
                    checkRnf(choice_trialLoop)  # check whether to add to credits
            elif resp == corrResp2:
                nResp2_this_trial += 1
                credits -= float(expInfo['cost_response'])  # subtract response cost
                if cond_val2 == 'val':
                    nResp_since_last_rnf += 1
                    checkRnf(choice_trialLoop)  # check whether to add to credits
            # record resp and resp time for csv
            choice_trialLoop.addData("event", "choice key resp")
            # choice_trialLoop.addData("key_press", key_resp.keys)
            choice_trialLoop.addData("resp", resp)
            choice_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
            choice_trialLoop.addData("routineClock_t", str(t))
            for nframes in range(5):
                triangle_resp.setOpacity(1)
                win.flip()
            thisExp.nextEntry()  # it does not call giveRnf(), so needs to go to next line in csv here

    # elif phase == "contingency_test":
    #     if key_resp.keys in [corrResp1, corrResp2]:
    #         nResp += 1
    #         print("Nresp: {:d}".format(nResp))
    #         if key_resp.keys == corrResp:   # give reward if correct answer
    #             key_resp.corr = 1
    #             credits += float(expInfo['credits_per_rnf'])
    #         else:
    #             key_resp.corr = 0
    #         # record resp and resp time for csv
    #         contingency_trialLoop.addData("event", "contingency key resp")
    #         contingency_trialLoop.addData("key_press", key_resp.keys)
    #         contingency_trialLoop.addData("resp", key_resp_map[key_resp.keys])
    #         contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    #         contingency_trialLoop.addData("routineClock_t", str(t))
    #         thisExp.nextEntry()


# conditions list used to define create trial loops with TrialHandler
conditionsList = CounterBalance(expInfo['participant'], balancingList)

# Initialise credits
credits = 0.0
# Initialise some variables
nReinforcers, nResp = 0, 0
nResp1, nResp2 = 0, 0

# thresholds for joystick response
joy_resp_thresh = 0.75
joy_reset_thresh = 0.1
# setup joystick
joystick.backend = 'pyglet'  # must match the Window created
nJoys = joystick.getNumJoysticks()  # to check if we have any joysticks connected
id = 0
joy = joystick.Joystick(id)  # id must be <= nJoys - 1

# mouse event listener for consumption test
mouse = event.Mouse(win=win)

### Initialize text, image components used in main experiment

# TextStims for messages to participant
continue_msg = visual.TextStim(
    win=win, name='continue_msg',
    text="(Click anywhere to continue)",
    font='Arial', alignText='center',
    pos=(0, -0.8), height=0.06, wrapWidth=None, ori=0,
    color='dimgray', colorSpace='rgb', opacity=1,
    depth=0.0)
countdown_msg = visual.TextStim(
    win=win, name='countdown_msg',
    text="Starting in",
    font='Arial', alignText='center',
    pos=(0, -0.75), height=0.1, wrapWidth=None, ori=0,
    color='dimgray', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize TextStims and ImageStims for instructions

instr0a = "Coin Collector Game"
instr0a_txt = visual.TextStim(
    win=win, name='instr0a_txt',
    text=instr0a,
    font='Arial', alignText='center',
    pos=(0, 0.2), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr0b = "Welcome!\n\nThe following instructions will introduce you to the game."
instr0b_txt = visual.TextStim(
    win=win, name='instr0b_txt',
    text=instr0b,
    font='Arial', alignText='center',
    pos=(0, -0.1), height=0.06, wrapWidth=0.75 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-1.0)

instr1 = "In this experiment, you will have the chance to collect some coins.\n\n\n\n\n\n\n\n" \
         "Each coin collected is worth 40 CREDITS.\n" \
         "Gold and silver coins have the same value."
instr1_txt = visual.TextStim(
    win=win, name='instr1_txt',
    text=instr1,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.45 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
silver_coin_img = visual.ImageStim(
    win=win, name='silver_coin_img',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=[-0.15, 0.0], size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
gold_coin_img = visual.ImageStim(
    win=win, name='gold_coin_img',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=[0.15, 0.0], size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

instr2 = "The number of credits you have at the end of the experiment will be used to calculate a monetary bonus.\n\n" \
         "The bonus will be added to your payment for this experiment."
instr2_txt = visual.TextStim(
    win=win, name='instr2_txt',
    text=instr2,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.65 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr3 = "During the task, you will be allowed to perform one of two actions with the joystick:\n" \
         "    -   tilt left, or \n" \
         "    -   tilt right\n\n" \
         "You will know which action is allowed by a figure that will appear on the screen.\n\n" \
         "Each allowed action has a cost of 1 CREDIT."
instr_3_txt = visual.TextStim(
    win=win, name='instr_3_txt',
    text=instr3,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.85 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr4a = "When you see this figure, you will be allowed to respond by tilting the joystick to the right.\n" \
          "Try it out!"
instr_4a_txt = visual.TextStim(
    win=win, name='instr_4a_txt',
    text=instr4a,
    font='Arial', alignText='left',
    pos=(0, 0.4), height=0.06, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr4b = "Every time you perform the correct action for the figure, a triangle will appear at the top-right of the screen signalling that a response has been made.\n\n" \
          "(Keep practicing, or click anywhere to continue)"
instr_4b_txt = visual.TextStim(
    win=win, name='instr_4b_txt',
    text=instr4b,
    font='Arial', alignText='left',
    pos=(0, -0.45), height=0.06, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-1.0)
stim1_img = visual.ImageStim(
    win=win, name='stim1_img',
    image='stim' + os.sep + 'right_resp_img.png', mask=None,
    ori=0, pos=(-0.5, 0), size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

instr5a = "When you see this figure, you will be allowed to respond by tilting the joystick to the left.\n" \
          "Try it out!"
instr_5a_txt = visual.TextStim(
    win=win, name='instr_5a_txt',
    text=instr5a,
    font='Arial', alignText='left',
    pos=(0, 0.4), height=0.06, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr5b = "(Keep practicing, or click anywhere to continue)"
instr_5b_txt = visual.TextStim(
    win=win, name='instr_5_txt',
    text=instr5b,
    font='Arial', alignText='left',
    pos=(0, -0.4), height=0.06, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
stim2_img = visual.ImageStim(
    win=win, name='stim2_img',
    image='stim' + os.sep + 'left_resp_img.png', mask=None,
    ori=0, pos=(0.5, 0), size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

instr6 = "Your joystick actions will *sometimes* be rewarded with a coin (either silver or gold).\n\n" \
         "IMPORTANT: One of the actions (eg., left) will give you gold coins when you are rewarded; the other action (eg., right) will give you silver coins when you are rewarded.\n" \
         "This will remain the same throughout the whole experiment."
instr_6_txt = visual.TextStim(
    win=win, name='instr_6_txt',
    text=instr6,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.65 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr7 = "The coin (either silver or gold) at the center of the screen will light up every time you are rewarded for an action."
instr_7_txt = visual.TextStim(
    win=win, name='instr_7_txt',
    text=instr7,
    font='Arial', alignText='left',
    pos=(0, 0.4), height=0.06, wrapWidth=0.6 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
light_up_coins_img = visual.ImageStim(
    win=win, name='light_up_coins_img',
    image='stim' + os.sep + 'coins_light_up_instructions.png', mask=None,
    ori=0, pos=(0, -0.05), size=(0.75, 0.4), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)

instr8 = "There are 2 piggy banks:\n\n\n\n\n\n\n\n\n" \
         "Rewarded coins are stored in their corresponding piggy banks, provided they are not full.\n" \
         "A coin deposited into a piggy bank earns you 40 CREDITS.\n\n" \
         "You will be notified if a piggy bank becomes full."
instr_8_txt = visual.TextStim(
    win=win, name='instr_8_txt',
    text=instr8,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.65 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);
silver_piggy_img = visual.ImageStim(
    win=win, name='silver_piggy_img',
    image='stim' + os.sep + 'silver_piggy_bank_english.png', mask=None,
    ori=0, pos=(-0.2, 0.075), size=(0.2, 0.18), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
gold_piggy_img = visual.ImageStim(
    win=win, name='gold_piggy_img',
    image='stim' + os.sep + 'gold_piggy_bank_english.png', mask=None,
    ori=0, pos=(0.2, 0.075), size=(0.2, 0.18), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)

instr9 = "Given the nature of the task, not every correct joystick action you make will reward you with a coin.\n\n" \
         "It is your task to discover the best way of performing the actions to earn as many credits as possible during the experiment.\n\n" \
         "The best strategy will be the same for both actions, since they only differ in whether they give you silver or gold coins."
instr_9_txt = visual.TextStim(
    win=win, name='instr_9_txt',
    text=instr9,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.75 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr10 = "During the FREE COLLECTION phase of the experiment, you will be given 15 seconds to collect coins from the screen using your mouse.\n\n" \
          "Coins you pick up in the FREE COLLECTION phase will also be deposited into their respective piggy banks for 40 CREDITS each (provided the piggy bank is not full)."
instr_10_txt = visual.TextStim(
    win=win, name='instr_10_txt',
    text=instr10,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr11 = "During the CURTAIN phase of the experiment, you will be given 90 seconds to again make responses using the joystick.\n\n" \
          "This time, the coin at the center of the screen will be covered with this curtain:\n\n\n\n\n\n\n\n\n" \
          "and you will *not* see the red triangle when you make a response.\n\n" \
          "Apart from this, nothing else about the game has changed. \n" \
          "Allowed actions will still be denoted by their corresponding figures, " \
          "each action will still be associated with the same type of coin, " \
          "and awarded coins will be deposited in their respective piggy banks for 40 CREDITS (provided the piggy bank is not full)."
instr_11_txt = visual.TextStim(
    win=win, name='instr_11_txt',
    text=instr11,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr12 = "In summary:\n\n" \
          "1) The figure(s) shown tells you what action(s) is allowed.\n\n" \
          "2) You can perform the allowed actions at any time and as many times as you want while the corresponding figure is on the screen.\n\n" \
          "3) One of the two possible actions will sometimes give you gold coins, the other action will sometimes give you silver coins.\n\n" \
          "4) Each time you are rewarded with a coin, the coin is deposited in the appropriate piggy bank, provided it is not full.\n\n" \
          "5) 40 CREDITS will be added to your earnings whenever a coin is deposited into a piggy bank. Each time you perform an action, 1 CREDIT will be subtracted from your earnings.\n\n" \
          "Take into account that the experiment is designed so that you do not lose money. Therefore, it is to your advantage not to stop responding altogether during the experiment.\n\n\n" \
          "You are done with the instructions. \nPlease call the experimenter if you have any questions."
instr_12_txt = visual.TextStim(
    win=win, name='instr_12_txt',
    text=instr12,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

# TextStims for practice
welcome_txt = visual.TextStim(
    win=win, name='welcome_txt',
    text="Let's get started!",
    font='Arial', alignText='center',
    pos=(0, 0.3), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
begin_txt = visual.TextStim(
    win=win, name='begin_txt',
    text="Click anywhere to enter the experiment.",
    font='Arial', alignText='center',
    pos=(0, -0.1), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

training_ready_txt = visual.TextStim(
    win=win, name='instr_buttondown_txt',
    text="Use the joystick to make responses.",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=0.75 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "training"
triangle_resp = visual.ShapeStim(  # also used for Routine "choice_test"
    win=win, name='triangle_resp',
    vertices=[[-(0.2, 0.2)[0] / 2.0, -(0.2, 0.2)[1] / 2.0], [+(0.2, 0.2)[0] / 2.0, -(0.2, 0.2)[1] / 2.0],
              [0, (0.2, 0.2)[1] / 2.0]],
    ori=0, pos=(0.85, 0.85),
    lineWidth=2, lineColor=[1.000, 0.004, -1.000], lineColorSpace='rgb',
    fillColor=[1.000, -1.000, -1.000], fillColorSpace='rgb',
    opacity=1.0, depth=-1.0, interpolate=True)
stim = visual.ImageStim(
    win=win, name='stim',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
coin = visual.ImageStim(
    win=win, name='coin',
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)

# Create some text box objects for debugging purposes; if debugging == True these will show up
num_credits_txt = visual.TextBox(window=win, text=' ', font_size=50,
                                 font_color=[-1, -1, 1], size=(1.9, .3), pos=(0.7, -0.95),
                                 grid_horz_justification='center', units='norm')
credits_title_txt = visual.TextBox(window=win, text='Credits', font_size=80,
                                   font_color=[1, 1, 1], size=(1.9, .3), pos=(0.7, -0.8),
                                   grid_horz_justification='center', units='norm')

# Initialize components for Routine "piggy_bank_partially_full"
piggy_part_txt = visual.TextStim(
    win=win, name='piggy_part_txt',
    text="The piggy banks are getting full!",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);
# Initialize components for Routine "piggy_bank_full"
piggy_full = "The following piggy bank:\n\n\n\n\n\n" \
             "is now FULL.\n\n" \
             "No further coins can be deposited in this piggy bank."
piggy_full_txt = visual.TextStim(
    win=win, name='piggy_full_txt',
    text=piggy_full,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);
devalued_piggy_img = visual.ImageStim(
    win=win, name='devalued_piggy_img',
    image='sin', mask=None,
    ori=0, pos=(0, 0.075), size=(0.3, 0.25), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
silver_piggy_fname = 'stim' + os.sep + 'silver_piggy_bank_english.png'
gold_piggy_fname = 'stim' + os.sep + 'gold_piggy_bank_english.png'

free_collection_txt = visual.TextStim(
    win=win, name='free_collection_txt',
    text="FREE COLLECTION",
    font='Arial', alignText='center',
    pos=(0, 0.35), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_consumption = "Use your mouse to collect up to 10 coins.\n\n" \
                    "You get " + str(fixed_durations['consumption']) + " seconds."
# Initialize components for Routine "instr_before_consumption_test"
instr_consumption_txt = visual.TextStim(
    win=win, name='instr_consumption_txt',
    text=instr_consumption,
    font='Arial', alignText='center',
    pos=(0, -0.1), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

coin_positions = [(-0.1 * wh_ratio, 0.1),
                  (-0.35 * wh_ratio, 0.1),
                  (0.35 * wh_ratio, 0.35),
                  (0 * wh_ratio, 0.3),
                  (-0.25 * wh_ratio, 0.2),
                  (0.175 * wh_ratio, 0.35),
                  (0.3 * wh_ratio, 0.15),
                  (-0.4 * wh_ratio, 0.3),
                  (0.15 * wh_ratio, 0.1),
                  (-0.15 * wh_ratio, 0.35),
                  (0.05 * wh_ratio, -0.3),
                  (-0.3 * wh_ratio, -0.15),
                  (-0.05 * wh_ratio, -0.1),
                  (0.225 * wh_ratio, -0.35),
                  (-0.1 * wh_ratio, -0.3),
                  (0.15 * wh_ratio, -0.15),
                  (-0.4 * wh_ratio, -0.25),
                  (-0.225 * wh_ratio, -0.375),
                  (0.3 * wh_ratio, -0.1),
                  (0.4 * wh_ratio, -0.3)]
shuffle(coin_positions)

# Initialize 10 gold and 10 silver coins
gold1 = visual.ImageStim(
    win=win, name='gold1',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
gold2 = visual.ImageStim(
    win=win, name='gold2',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
gold3 = visual.ImageStim(
    win=win, name='gold3',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
gold4 = visual.ImageStim(
    win=win, name='gold4',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
gold5 = visual.ImageStim(
    win=win, name='gold5',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
gold6 = visual.ImageStim(
    win=win, name='gold6',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-6.0)
gold7 = visual.ImageStim(
    win=win, name='gold7',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-7.0)
gold8 = visual.ImageStim(
    win=win, name='gold8',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-8.0)
gold9 = visual.ImageStim(
    win=win, name='gold9',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-9.0)
gold10 = visual.ImageStim(
    win=win, name='gold10',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-10.0)
silver1 = visual.ImageStim(
    win=win, name='silver1',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-11.0)
silver2 = visual.ImageStim(
    win=win, name='silver2',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-12.0)
silver3 = visual.ImageStim(
    win=win, name='silver3',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-13.0)
silver4 = visual.ImageStim(
    win=win, name='silver4',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-14.0)
silver5 = visual.ImageStim(
    win=win, name='silver5',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-15.0)
silver6 = visual.ImageStim(
    win=win, name='silver6',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-16.0)
silver7 = visual.ImageStim(
    win=win, name='silver7',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-17.0)
silver8 = visual.ImageStim(
    win=win, name='silver8',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-18.0)
silver9 = visual.ImageStim(
    win=win, name='silver9',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-19.0)
silver10 = visual.ImageStim(
    win=win, name='silver10',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0,0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-20.0)

# Initialize components for Routine "instr_after_consumption_test"
instr_after_consumption_test = visual.TextStim(
    win=win, name='instr_after_consumption_test',
    text="OK, we will deposit these coins into their respective piggy banks.",
    font='Arial', alignText='left',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);

hidden_coin_txt = visual.TextStim(
    win=win, name='hidden_coin_txt',
    text="HIDDEN COIN",
    font='Arial', alignText='center',
    pos=(0, 0.4), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_choice_test_str = "Use the trackball to make responses.\n\n" \
                        "Apart from the curtain covering the coins, nothing about the game has changed.\n\n" \
                        "You will get " + str(fixed_durations['choice_test']) + " seconds.  Ready?"
# Initialize components for Routine "instr_choice_test"
instr_choice_test_txt = visual.TextStim(
    win=win, name='instr_choice_test_txt',
    text=instr_choice_test_str,
    font='Arial', alignText='left',
    pos=(0, -0.15), height=0.1, wrapWidth=0.75 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);

# Initialize components for Routine "choice_test"
stim_choice1 = visual.ImageStim(
    win=win, name='stim_choice1',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=(0.2, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
stim_choice_2 = visual.ImageStim(
    win=win, name='stim_choice_2',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=(0.2, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
curtain_img = visual.ImageStim(
    win=win, name='curtain_img',
    image='stim' + os.sep + 'curtain.png', mask=None,
    ori=0, pos=(0, 0.03), size=(0.2, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)

# Initialize components for Routine "instr_contingency_test"
instr_contingency_test = "You are done with the experiment!\n\n\n" \
                         "Now you will answer some questions about the task.\n\n" \
                         "Ready?"
instr_contingency_test_txt = visual.TextStim(
    win=win, name='text',
    text=instr_contingency_test,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);

# Initialize components for Routine "contingency_test"
coin_contingency_img = visual.ImageStim(
    win=win, name='coin_contingency_img',
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(0.3, 0.3), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
coin_contingency_txt = visual.TextStim(
    win=win, name='coin_contingency_txt',
    text="Perform the trackball roll that gave you this coin during training.",
    font='Arial', alignText='left',
    pos=(0, 0.6), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-3.0);
stim_contingency_img = visual.ImageStim(
    win=win, name='stim_contingency_img',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=(0.3, 0.3), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
stim_contingency_txt = visual.TextStim(
    win=win, name='stim_contingency_txt',
    text="Perform the trackball roll associated with this figure.",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-3.0);

# Initialize components for Routine "goodbye"
final_txt = visual.TextStim(
    win=win, name='final_txt',
    text="End of the experiment.\n\nThanks for participating!",
    font='Arial', alignText='left',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);
credits_txt = visual.TextStim(
    win=win, name='credits_txt',
    text='default text',
    font='Arial', alignText='left',
    pos=(0, -0.35), height=0.25, wrapWidth=None, ori=0,
    color='yellow', colorSpace='rgb', opacity=1,
    depth=-1.0);

if not debugging:
    win.mouseVisible = False

####################   BEGIN INSTRUCTIONS   #####################

show_instr([instr0a_txt, instr0b_txt], continue_msg, routineClock, win, mouse, 0)
show_instr([instr1_txt, silver_coin_img, gold_coin_img], continue_msg, routineClock, win, mouse, 0)
show_instr([instr2_txt], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_3_txt], continue_msg, routineClock, win, mouse, 0)

# ------ start Routine "instr_4"-------
frameN = -1
continueRoutine = True
resp1_practComponents = [instr_4a_txt, stim1_img, triangle_resp]
instr_4b_txt.status = NOT_STARTED
set_components_attr(resp1_practComponents, 'status', STARTED)
set_components_attr(resp1_practComponents, 'autoDraw', True)
ready_for_resp = False
mouse_allowed = False
prev_buttons = mouse.getPressed()
while continueRoutine:
    # get current time
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    theseKeys = event.getKeys(keyList=['escape', 'q'])
    # check for quit:
    if theseKeys:
        core.quit()

    triangle_resp.setOpacity(0, log=False)

    joy_x = joy.getX()
    if ready_for_resp:
        if joy_x > joy_resp_thresh:
            for frameN in range(5):  # show for bit more than half a sec (60 frames = 1 sec)
                triangle_resp.setOpacity(1)
                win.flip()
            if instr_4b_txt.status == NOT_STARTED:
                instr_4b_txt.setAutoDraw('True')
                resp1_practComponents.append(instr_4b_txt)
                mouse_allowed = True  # after registering a joystick response, allow participant to continue
            ready_for_resp = False
    else:
        # ready to accept next response when position returns to center
        if abs(joy_x) < joy_reset_thresh:
            ready_for_resp = True

    if mouse_allowed:
        buttons = mouse.getPressed()
        if all([item == 0 for item in prev_buttons]):
            if any(buttons):
                # a response ends the routine
                set_components_attr(resp1_practComponents, 'status', FINISHED)
                set_components_attr(resp1_practComponents, 'autoDraw', False)
                continueRoutine = False
        prev_buttons = buttons
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    # refresh the screen
    win.flip()

# ------ start Routine "instr_5"-------
frameN = -1
continueRoutine = True
resp2_practComponents = [instr_5a_txt, stim2_img, triangle_resp]
instr_5b_txt.status = NOT_STARTED
set_components_attr(resp2_practComponents, 'autoDraw', True)
ready_for_resp = False
mouse_allowed = False
prev_buttons = mouse.getPressed()
while continueRoutine:
    # get current time
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    theseKeys = event.getKeys(keyList=['escape', 'q'])
    # check for quit:
    if theseKeys:
        core.quit()

    triangle_resp.setOpacity(0, log=False)

    joy_x = joy.getX()
    if ready_for_resp:
        if joy_x < -joy_resp_thresh:
            for frameN in range(5):  # show for bit more than half a sec (60 frames = 1 sec)
                triangle_resp.setOpacity(1)
                win.flip()
            if instr_5b_txt.status == NOT_STARTED:
                instr_5b_txt.setAutoDraw('True')
                resp2_practComponents.append(instr_5b_txt)
                mouse_allowed = True  # after registering a joystick response, allow participant to continue
            ready_for_resp = False
    else:
        # ready to accept next response when position returns to center
        if abs(joy_x) < joy_reset_thresh:
            ready_for_resp = True

    if mouse_allowed:
        buttons = mouse.getPressed()
        if all([item == 0 for item in prev_buttons]):
            if any(buttons):
                # a response ends the routine
                set_components_attr(resp2_practComponents, 'autoDraw', False)
                continueRoutine = False
        prev_buttons = buttons
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    # refresh the screen
    win.flip()

show_instr([instr_6_txt], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_7_txt, light_up_coins_img], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_8_txt, silver_piggy_img, gold_piggy_img], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_9_txt], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_10_txt], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_11_txt, curtain_img], continue_msg, routineClock, win, mouse, 0)
show_instr([instr_12_txt], None, routineClock, win, mouse, 0)

####################   END INSTRUCTIONS   #######################


####################   BEGIN EXPERIMENT   #######################
show_instr([welcome_txt, begin_txt], None, routineClock, win, mouse, 0)

nResp = 0  # reset response counter

trialList_consumption = conditionsList[2:4]
blockLoop = data.TrialHandler(nReps=1, method='sequential',
                              extraInfo=expInfo, originPath=-1,
                              trialList=trialList_consumption,
                              seed=None, name='blockLoop')
thisExp.addLoop(blockLoop)  # add the loop to the experiment
# abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)

for thisBlock in blockLoop:

    # for repN in range(nBlocks):
    currentLoop = blockLoop

    # set up handler to look after randomisation of conditions etc
    trialList_training = conditionsList[:2]
    trialLoop = data.TrialHandler(nReps=nTrials, method='sequential',
                                  extraInfo=expInfo, originPath=-1,
                                  trialList=trialList_training,
                                  seed=None, name='trialLoop')
    thisExp.addLoop(trialLoop)  # add the loop to the experiment

    for thisTrial in trialLoop:
        if trialLoop.thisN == 0:
            trialLoop.addData("event", "instr before block")
            trialLoop.addData("globalClock_t", str(globalClock.getTime()))
            thisExp.nextEntry()
            show_instr([training_ready_txt], continue_msg, routineClock, win, mouse, 0)

        currentLoop = trialLoop
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))

        # ------Prepare to start Routine "training"-------
        t = 0
        routineClock.reset()  # clock
        frameN = -1
        continueRoutine = True

        # update component parameters for each repeat
        stim.setPos(position)
        stim.setImage(stimulus)

        shouldBeReinforced = 0  # flag to set next reinforcer for RI schedule
        tickClock = core.CountdownTimer(1)  # this is to tick every second for reward availability in the RI schedule
        trials_this_rep, nResp_this_trial = 0, 0
        nResp_since_last_rnf = 0  # to track responses since last rnf and truncate distribution in case too many responses have been performed without a rnf being delivered
        nResp_until_next_rnf = round(min(
            expInfo['max_resp_bet_rnf'],
            max(
                expInfo['min_resp_bet_rnf'],
                np.random.normal(expInfo['mean_resp_bet_rnf'], expInfo['sigma_resp_bet_rnf'])
            )
        ))

        trial_duration = expInfo['trial_duration']
        print("trial dur: " + str(trial_duration))
        print("nResp until next rnf: " + str(nResp_until_next_rnf))

        # add entry for training start
        trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        trialLoop.addData("routineClock_t", str(t))
        trialLoop.addData("trial", nBlocks * nTrials * blockLoop.thisN + trialLoop.thisN)
        trialLoop.addData("event", "training period start")
        trialLoop.addData("duration", trial_duration)
        trialLoop.addData("nResp_untilRnf", nResp_until_next_rnf)
        thisExp.nextEntry()

        coin.setImage(coin_img)
        # keep track of which components have finished
        trainingComponents = [triangle_resp, stim, coin]
        set_components_attr(trainingComponents, 'status', NOT_STARTED)
        ready_for_resp = False

        # -------Start Routine "training"-------
        while continueRoutine:
            # get current time
            t = routineClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame

            theseKeys = event.getKeys(keyList=['escape', 'q'])
            # check for quit:
            if theseKeys:
                core.quit()
            # show debugging stuff?
            if debugging:
                show_debugging_stuff()

            for component in trainingComponents:
                if t >= 0.0 and component.status == NOT_STARTED:
                    component.setAutoDraw(True)
                    event.clearEvents(eventType='keyboard')
            if triangle_resp.status == STARTED:  # only update if drawing
                triangle_resp.setOpacity(0, log=False)
            if coin.status == STARTED:  # only update if drawing
                coin.setOpacity(0.2, log=False)
            theseKeys = event.getKeys(keyList=['escape', 'q'])

            framesRemain = 0.0 + trial_duration - 20 * win.monitorFramePeriod
            if t <= framesRemain:
                joy_x = joy.getX()
                if ready_for_resp:
                    if joy_x > joy_resp_thresh:
                        resp_made('right')
                        ready_for_resp = False
                    if joy_x < -joy_resp_thresh:
                        resp_made('left')
                        ready_for_resp = False
                else:
                    # ready to accept next response when position returns to center
                    if abs(joy_x) < joy_reset_thresh:
                        ready_for_resp = True
            else:
                continueRoutine = False

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                set_components_attr(trainingComponents, 'autoDraw', False)
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        # -------Ending Routine "training"-------

    if blockLoop.thisTrialN == 0:
        # ------ Start Routine "piggy_bank_partially_full"-------
        # add entry for piggy message start
        thisExp.addData("globalClock_t", str(globalClock.getTime()))
        thisExp.addData("event", "piggy getting full msg")
        thisExp.nextEntry()
        show_instr([piggy_part_txt], continue_msg, routineClock, win, mouse, 2)
        # ------ End Routine "piggy_bank_partially_full"-------
    else:
        # ------Start Routine "piggy_bank_full"-------
        devalued_piggy_img.setImage(silver_piggy_fname if devalued_coin == "silver" else gold_piggy_fname)
        piggy_msg_duration = fixed_durations['piggy_bank_full']
        routineTimer.reset(piggy_msg_duration)
        # add entry for piggy message start
        thisExp.addData("globalClock_t", str(globalClock.getTime()))
        thisExp.addData("event", "piggy full msg")
        thisExp.nextEntry()
        show_instr([piggy_full_txt, devalued_piggy_img], continue_msg, routineClock, win, mouse, 2)
        # ------End Routine "piggy_bank_full"-------

    # ------Prepare to start Routine "instr_before_consumption_test"-------
    thisExp.addData("globalClock_t", globalClock.getTime())
    thisExp.addData("event", 'instr before consumption test')
    thisExp.nextEntry()
    show_instr([free_collection_txt, instr_consumption_txt], continue_msg, routineClock, win, mouse, 2)

    # ------Prepare to start Routine "consumption_test"-------
    # mouse.setPos((0, 0))
    # win.mouseVisible=True
    gold_coins = [gold1, gold2, gold3, gold4, gold5, gold6, gold7, gold8, gold9, gold10]
    silver_coins = [silver1, silver2, silver3, silver4, silver5, silver6, silver7, silver8, silver9, silver10]
    selected_coins = []

    t = 0
    routineClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # get conditions for consumption test
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))

    prev_buttons = mouse.getPressed()

    consumption_duration = fixed_durations['consumption']
    blockLoop.addData("globalClock_t", globalClock.getTime())
    blockLoop.addData("routineClock_t", str(t))
    blockLoop.addData("event", 'consumption test start')
    blockLoop.addData("duration", consumption_duration)
    thisExp.nextEntry()

    # routineTimer.reset(consumption_duration)
    consumption_testComponents = gold_coins + silver_coins
    set_components_attr(consumption_testComponents, 'status', NOT_STARTED)

    delay_start_t = 0
    # -------Start Routine "consumption_test"-------
    while continueRoutine:
        # get current time
        # t_remain = routineTimer.getTime()
        t = routineClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)

        theseKeys = event.getKeys(keyList=['escape', 'q'])
        # check for quit:
        if theseKeys:
            core.quit()
        # show debugging stuff?
        if debugging:
            show_debugging_stuff()

        frameRemains = 0.0 + consumption_duration - win.monitorFramePeriod * 0.75  # most of one frame period left
        if t < frameRemains:
            for idx in range(len(consumption_testComponents)):
                component = consumption_testComponents[idx]
                if t < frameRemains:
                    if component.status == NOT_STARTED:
                        component.setOpacity(1, log=False)
                        component.setAutoDraw(True)
                        component.setPos(coin_positions[idx])
                        event.clearEvents(eventType='keyboard')
                    elif component.status == STARTED:
                        if len(selected_coins) < 10:
                            if mouse.isPressedIn(component):
                                selected = consumption_testComponents.pop(idx)
                                if selected in gold_coins:
                                    resp_made("gold")
                                if selected in silver_coins:
                                    resp_made("silver")
                                selected.setOpacity(0)
                                selected_coins.append(selected)
                                break
                        else:
                            component.setOpacity(0.2)
                            if not delay_start_t:
                                delay_start_t = t
                            else:
                                if t > delay_start_t + 0.1:
                                    component.setAutoDraw(False)
                else:
                    if component.status == STARTED:
                        component.setAutoDraw(False)
        else:
            set_components_attr(consumption_testComponents, 'autoDraw', False)
            continueRoutine = False
        continueRoutine = False  # will revert to True if at least one component still running
        for component in consumption_testComponents:
            if hasattr(component, "status") and component.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        # refresh the screen
        win.flip()
    # -------Ending Routine "consumption_test"-------

    # ------Prepare to start Routine "instr_after_consumption_test"-------
    consumption_after_duration = fixed_durations['instr_after_consumption_test']
    blockLoop.addData("globalClock_t", str(globalClock.getTime()))
    blockLoop.addData("routineClock_t", str(t))
    blockLoop.addData("event", "instr after consumption test")
    blockLoop.addData("duration", consumption_after_duration)
    thisExp.nextEntry()
    # keep track of which components have finished
    routineTimer.reset(consumption_after_duration)  # clock
    instr_after_consumption_testComponents = [instr_after_consumption_test]
    show_timed_countdown(instr_after_consumption_testComponents, None, routineTimer, win)
    # -------Ending Routine "instr_after_consumption_test"-------

    # ------Prepare to start Routine "rest_after_block"-------'
    after_block_duration = fixed_durations['after_block']
    thisExp.addData("globalClock_t", str(globalClock.getTime()))
    thisExp.addData("event", "rest after block")
    thisExp.addData("duration", after_block_duration)
    thisExp.nextEntry()
    routineTimer.reset(after_block_duration)  # clock
    show_timed_countdown([], None, routineTimer, win)
    # -------Ending Routine "rest_after_block"-------
# completed nBlocks repeats of 'blockLoop'


# set up handler to look after randomisation of conditions etc
choice_trialLoop = data.TrialHandler(nReps=1, method='sequential',
                                     extraInfo=expInfo, originPath=-1,
                                     trialList=[conditionsList[4]],
                                     seed=None, name='choice_trialLoop')
thisExp.addLoop(choice_trialLoop)  # add the loop to the experiment

for thisTrial in choice_trialLoop:
    currentLoop = choice_trialLoop
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
            # if expInfo['hand'] == 'left':
            #     if paramName in ["corrResp1", "corrResp2"]:
            #         exec('{} = key_swap_map[thisTrial[paramName]]'.format(paramName))

    # ------Prepare to start Routine "instr_choice_test"-------
    instr_choice_test_duration = fixed_durations['instr_choice_test']
    thisExp.addData("globalClock_t", str(globalClock.getTime()))
    thisExp.addData("event", "instructions before choice test")
    thisExp.addData("duration", instr_choice_test_duration)
    thisExp.nextEntry()
    routineTimer.reset(instr_choice_test_duration)  # clock
    show_timed_countdown([hidden_coin_txt, instr_choice_test_txt], countdown_msg, routineTimer, win,
                         countdown_t_minus=5)

    # ------Prepare to start Routine "choice_test"-------
    t = 0
    routineClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    curtain_img.setPos((0, 0))  # curtain was shifted up for instructions, so move it back to center
    stim_choice1.setPos(position1)
    stim_choice1.setImage(stimulus_choice1)
    stim_choice_2.setPos(position2)
    stim_choice_2.setImage(stimulus_choice2)

    shouldBeReinforced = 0  # just in case not running the training stage for debugging purposes
    # tickClock = core.CountdownTimer(1)  # same reason as line above

    shouldBeReinforced_1, shouldBeReinforced_2 = 0, 0  # set rnf flag for two options
    # this is to tick every second for reward availability
    # tickClock1, tickClock2 = core.CountdownTimer(1), core.CountdownTimer(1)

    nResp1, nResp2 = 0, 0  # to track responses in this trial;
    nResp_since_last_rnf = 0  # to track responses since last rnf and truncate distribution in case too many responses have been performed without a rnf being delivered
    nResp_until_next_rnf = round(min(
        expInfo['max_resp_bet_rnf'],
        max(
            expInfo['min_resp_bet_rnf'],
            np.random.normal(expInfo['mean_resp_bet_rnf'], expInfo['sigma_resp_bet_rnf'])
        )
    ))
    choice_duration = fixed_durations['choice_test']
    choice_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    choice_trialLoop.addData("routineClock_t", str(t))
    choice_trialLoop.addData("event", 'choice test start')
    choice_trialLoop.addData("duration", choice_duration)
    choice_trialLoop.addData("nResp_untilRnf", nResp_until_next_rnf)
    thisExp.nextEntry()

    # keep track of which components have finished
    choice_testComponents = [triangle_resp, stim_choice1, stim_choice_2, curtain_img]
    set_components_attr(choice_testComponents, 'status', NOT_STARTED)
    mouse.setPos((0, 0))

    # -------Start Routine "choice_test"-------
    while continueRoutine:
        # get current time
        t = routineClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        theseKeys = event.getKeys(keyList=['escape', 'q'])
        # check for quit:
        if theseKeys:
            core.quit()
        # show debugging stuff?
        if debugging:
            show_debugging_stuff()

        frameRemains = 0.0 + choice_duration - win.monitorFramePeriod * 0.75  # most of one frame period left

        # *triangle_resp* updates
        for component in choice_testComponents:
            if t >= 0.0 and component.status == NOT_STARTED:
                component.setAutoDraw(True)
                event.clearEvents(eventType='keyboard')
            if component.status == STARTED and t >= frameRemains:
                component.setAutoDraw(False)
        if triangle_resp.status == STARTED:  # only update if drawing
            triangle_resp.setOpacity(0, log=False)

        (x, y) = mouse.getPos()
        if x > mouse_thresh:
            if expInfo['hand'] == 'right':
                resp_made('down')
            else:
                resp_made('up')
        if x < -mouse_thresh:
            if expInfo['hand'] == 'right':
                resp_made('up')
            else:
                resp_made('down')

        # check if all components have finished
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in choice_testComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
# -------Ending Routine "choice_test"-------


# set up handler to look after randomisation of conditions etc
contingency_trialLoop = data.TrialHandler(nReps=2, method='random',
                                          extraInfo=expInfo, originPath=-1,
                                          trialList=conditionsList[:2],
                                          seed=None, name='contingency_trialLoop')
thisExp.addLoop(contingency_trialLoop)  # add the loop to the experiment
for thisTrial in contingency_trialLoop:
    currentLoop = contingency_trialLoop
    # get condition variables from training phase
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    # set phase to contingency
    phase = 'contingency'

    if contingency_trialLoop.thisN == 0:
        # ------Prepare to start Routine "instr_contingency_test"-------
        instr_contingency_test_duration = fixed_durations['instr_contingency_test']
        contingency_trialLoop.addData("routineClock_t", str(t))
        contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        contingency_trialLoop.addData("event", "instructions before contingency test")
        contingency_trialLoop.addData("duration", instr_contingency_test_duration)
        thisExp.nextEntry()
        routineTimer.reset(instr_contingency_test_duration)  # clock
        # show_timed([], routineTimer, win, static_period)
        instr_contingency_testComponents = [instr_contingency_test_txt]
        show_timed_countdown(instr_contingency_testComponents, countdown_msg, routineTimer, win, countdown_t_minus=5)

    # ------Prepare to start Routine "contingency_test"-------

    # short pause before each question
    pause_duration = fixed_durations["pause_between_qs"]
    routineTimer.reset(pause_duration)
    contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    contingency_trialLoop.addData("event", "pause before contingency question")
    contingency_trialLoop.addData("duration", pause_duration)
    thisExp.nextEntry()
    show_timed_countdown([], None, routineTimer, win)

    t = 0
    routineClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    if contingency_trialLoop.thisRepN == 0:
        coin.setImage(coin_img)
        coin.setOpacity(1)
        contingency_test_img = coin
        contingency_test_txt = coin_contingency_txt
        contingency_trialLoop.addData("event", "contingency start (coin)")
    else:
        stim.setImage(stimulus)
        stim.setPos(position)
        contingency_test_img = stim
        contingency_test_txt = stim_contingency_txt
        contingency_trialLoop.addData("event", "contingency start (stim)")

    contingency_testComponents = [contingency_test_img, contingency_test_txt]
    nResp = 0
    contingency_test_duration = fixed_durations['contingency_test']
    contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    contingency_trialLoop.addData("routineClock_t", str(t))
    contingency_trialLoop.addData('duration', contingency_test_duration)
    thisExp.nextEntry()

    # contingency_testComponents = [coin_contingency_img, key_resp, coin_contingency_txt]
    set_components_attr(contingency_testComponents, 'status', NOT_STARTED)

    mouse.setPos((0, 0))
    # -------Start Routine "contingency_test"-------
    while continueRoutine:
        # get current time
        t = routineClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        theseKeys = event.getKeys(keyList=['escape', 'q'])
        # check for quit:
        if theseKeys:
            core.quit()
        # show debugging stuff?
        if debugging:
            show_debugging_stuff()

        for component in contingency_testComponents:
            if component.status == NOT_STARTED:
                component.setAutoDraw(True)
                event.clearEvents(eventType='keyboard')

        (x, y) = mouse.getPos()
        if x > mouse_thresh:
            if expInfo['hand'] == 'right':
                resp_made('down')
            else:
                resp_made('up')
            continueRoutine = False
        if x < -mouse_thresh:
            if expInfo['hand'] == 'right':
                resp_made('up')
            else:
                resp_made('down')
            continueRoutine = False

        frameRemains = 0.0 + contingency_test_duration - win.monitorFramePeriod * 0.75  # most of one frame period left
        if t >= frameRemains:
            continueRoutine = False

        if not continueRoutine:
            set_components_attr(contingency_testComponents, 'autoDraw', False)

        # refresh the screen
        win.flip()

# ------Prepare to start Routine "goodbye"-------
credits_txt.setText(str(int(credits)) + ' credits')

# keep track of which components have finished
# add entry for goodbye start
goodbye_duration = fixed_durations['goodbye']

thisExp.addData("globalClock_t", str(globalClock.getTime()))
thisExp.addData("routineClock_t", str(t))
thisExp.addData("event", "goodbye message start")
thisExp.addData("duration", goodbye_duration)
thisExp.nextEntry()

goodbyeComponents = [final_txt, credits_txt]
routineTimer.reset(goodbye_duration)
show_timed_countdown(goodbyeComponents, None, routineTimer, win)

thisExp.addData("globalClock_t", str(globalClock.getTime()))
# thisExp.addData("routineClock_t", str(t))
thisExp.addData("event", "goodbye message end")

# compute and display payout for the session
payout_this_session = (nReinforcers * float(expInfo['credits_per_rnf']) - (nResp + nResp1 + nResp2) * float(
    expInfo['cost_response'])) / 100
print("payout: " + str(payout_this_session))

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename + '.csv')
thisExp.saveAsPickle(filename)

# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
