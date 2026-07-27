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
from psychopy.hardware import forp
from psychopy.hardware.emulator import ResponseEmulator, launchScan
from psychopy import visual

# Import helper functions
from helper_functions import *

# experiment mode
# Should we show the debugging msgs on screen?
# In the present experiment, this means showing the credits on the screen
debugging = False
emulator_mode = 'Scan'

# Ensure that relative paths start from the same directory as this script
_thisDir = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(str(_thisDir))

# Store info about the experiment
if debugging:
    nRepeat = 3
    min_ITI = 5
    max_ITI = 10
    min_trial_duration = 15
    max_trial_duration = 30
else:
    nRepeat = 5
    min_ITI = 15
    max_ITI = 30
    min_trial_duration = 30
    max_trial_duration = 60

# get next subject number (lowest number not already in DATA directory)
data_path = _thisDir.parent.joinpath("DATA" + os.sep + "fMRI").resolve()
if os.path.isdir(str(data_path)) == False:
    os.mkdir(str(data_path))

subdirs = [o for o in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, o))]
subj_numbers = [int(s.split('-')[-1]) for s in subdirs]
subj_id = 1
while subj_id in subj_numbers:
    subj_id += 1
if debugging:
    subj_id = 999

expName = 'habits_v_MRI'  # from the Builder filename that created this script
expInfo = {'participant': subj_id,
           'task_ordering': 1,  # 'age': 18, 'gender': 'm', 'ethnicity': '',
           'hand': 'right'
           }

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['n_repeat_training'] = nRepeat
expInfo['credits_per_rnf'] = 40
expInfo['cost_response'] = 1
expInfo['min_resp_bet_rnf'] = 5
expInfo['max_resp_bet_rnf'] = 30
expInfo['mean_resp_bet_rnf'] = 15
expInfo['sigma_resp_bet_rnf'] = 5
expInfo['min_ITI'] = min_ITI
expInfo['max_ITI'] = max_ITI
expInfo['min_trial_duration'] = min_trial_duration
expInfo['max_trial_duration'] = max_trial_duration
expInfo['schedule'] = 'RR'

# set data paths
part_path = data_path.joinpath('sub-{}'.format(expInfo['participant']))

# # get path for instructions and adapt text for experiment
# text_path = _thisDir.joinpath('text', 'english')

# set condition balancing list
balancingList = 'ConditionsLists_habits_fMRI.xlsx'

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

fullscreen = not debugging
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

# StaticPeriod for showing still instruction slides
# static_period = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='instruction_SP')

# define durations of instructions, trials, ITIs etc
ITI_durations = linspace(expInfo['min_ITI'], expInfo['max_ITI'],
                         nBlocks * nTrials)  # ITI durations linspace for each scanner run
ITI_durations = [list(np.random.permutation(ITI_durations)), list(np.random.permutation(ITI_durations))]

trial_durations = linspace(expInfo['min_trial_duration'], expInfo['max_trial_duration'], nTrials)
trial_durations = [[list(np.random.permutation(trial_durations)), list(np.random.permutation(trial_durations))] for idx
                   in range(nBlocks)]

fixed_durations = {
    "practice_answer": 1.5,
    "training_ready": 6,
    "piggy_bank_partially_full": 7,
    "piggy_bank_full": 7,
    "consumption": 10 if debugging else 15,
    "instr_before_consumption_test": 7,
    "instr_after_consumption_test": 3,
    "after_block": 3,
    "instr_choice_test": 10,
    "choice_test": 30 if debugging else 90,
    "instr_contingency_test": 7,
    "pause_between_qs": 1,
    "goodbye": 3
}
run1_duration = float(fixed_durations['training_ready'] + \
                      sum(ITI_durations[0]) + sum(sum(l) for l in trial_durations[0]) +
                      fixed_durations['piggy_bank_partially_full'] + \
                      fixed_durations['instr_before_consumption_test'] + fixed_durations['consumption'] +
                      fixed_durations[
                          'instr_after_consumption_test'])
run2_duration = float(fixed_durations['training_ready'] + \
                      sum(ITI_durations[1]) + sum(sum(l) for l in trial_durations[1]) + \
                      fixed_durations['piggy_bank_full'] + \
                      fixed_durations['instr_before_consumption_test'] + fixed_durations['consumption'] +
                      fixed_durations[
                          'instr_after_consumption_test']) + fixed_durations['after_block'] + \
                fixed_durations['instr_choice_test'] + fixed_durations['choice_test']
TR = 1.000
run1_volumes = np.ceil(run1_duration / TR)
run2_volumes = np.ceil(run2_duration / TR)
# settings for launchScan:
MR_settings = {
    'TR': TR,  # duration (sec) per whole-brain volume
    'volumes': None,  # number of whole-brain 3D volumes per scanning run
    'sync': '5',  # character to use as the sync timing event; assumed to come at start of a volume
    'skip': 0,  # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
    'sound': True  # in test mode: play a tone as a reminder of scanner noise
}


# # add MR settings to expInfo
# expInfo.update(MR_settings)

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

    if shouldBeReinforced:
        nResp_since_last_rnf = 0  # reset after reinforcer is delivered
        nResp_until_next_rnf = round(min(
            expInfo['max_resp_bet_rnf'],
            max(expInfo['min_resp_bet_rnf'],
                np.random.normal(expInfo['mean_resp_bet_rnf'], expInfo['sigma_resp_bet_rnf']))
        ))
        loop.addData("nResp_untilRnf", nResp_until_next_rnf)
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
    global credits, nResp, nResp_this_trial, nResp1, nResp2, nResp_since_last_rnf  # credits var updates throughout the experiment
    ''' after a resp is made, record it for the csv file and other actions for current phase '''

    if phase == "practice":
        if resp in [corrResp]:
            nResp += 1
            for nframes in range(5):  # show triangle
                triangle_resp.setOpacity(1)
                win.flip()
        mouse.setPos((0, 0))

    elif phase == "training":
        # record resp and resp time for csv
        trialLoop.addData("event", "training_resp")
        trialLoop.addData("resp", resp)
        trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        trialLoop.addData("routineClock_t", str(t))
        trialLoop.addData("trial", nBlocks * nTrials * blockLoop.thisN + trialLoop.thisN)
        trialLoop.addData("run", blockLoop.thisTrialN)
        if resp in [corrResp]:
            # if valid response for stimulus
            nResp += 1
            nResp_since_last_rnf += 1
            credits -= float(expInfo['cost_response'])  # subtract cost of response from credits
            trialLoop.addData("selected_coin", outcome)
            for nframes in range(5):  # show triangle
                triangle_resp.setOpacity(1)
                win.flip()
            checkRnf(trialLoop)  # check whether to reinforce
        thisExp.nextEntry()  # go to next entry
        mouse.setPos((0, 0))

    elif phase == "consumption":
        if resp in [corrResp1, corrResp2]:
            # add to credits if response is valid and is valued
            if resp == corrResp1:
                if cond_val1 == 'val':
                    credits += float(expInfo['credits_per_rnf'])
                blockLoop.addData("selected_coin", outcome_choice1)
            elif resp == corrResp2:
                if cond_val2 == 'val':
                    credits += float(expInfo['credits_per_rnf'])
                blockLoop.addData("selected_coin", outcome_choice2)
            # record resp and resp time for csv
            blockLoop.addData("event", "consumption_resp")
            blockLoop.addData("resp", resp)
            blockLoop.addData("globalClock_t", str(globalClock.getTime()))
            blockLoop.addData("routineClock_t", str(t))
            blockLoop.addData("run", blockLoop.thisTrialN)
            thisExp.nextEntry()

    elif phase == "choice":
        if resp in [corrResp1, corrResp2]:
            credits -= float(expInfo['cost_response'])  # subtract response cost
            if resp == corrResp1:
                nResp1 += 1
                if cond_val1 == 'val':
                    nResp_since_last_rnf += 1
                    checkRnf(choice_trialLoop)  # check whether to add to credits
                choice_trialLoop.addData("selected_coin", outcome_choice1)
            elif resp == corrResp2:
                nResp2 += 1
                if cond_val2 == 'val':
                    nResp_since_last_rnf += 1
                    checkRnf(choice_trialLoop)  # check whether to add to credits
                choice_trialLoop.addData("selected_coin", outcome_choice2)
            # record resp and resp time for csv
            choice_trialLoop.addData("event", "choice_resp")
            choice_trialLoop.addData("resp", resp)
            choice_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
            choice_trialLoop.addData("routineClock_t", str(t))
            choice_trialLoop.addData("run", blockLoop.nTotal-1)
            # for nframes in range(5):
            #     triangle_resp.setOpacity(1)
            #     win.flip()
            thisExp.nextEntry()  # it does not call giveRnf(), so needs to go to next line in csv here
        mouse.setPos((0, 0))

    elif phase == "contingency":
        if resp in [corrResp]:
            credits += float(expInfo['credits_per_rnf'])
        # record resp and resp time for csv
        contingency_trialLoop.addData("event", "contingency_resp")
        contingency_trialLoop.addData("resp", resp)
        contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        contingency_trialLoop.addData("routineClock_t", str(t))
        thisExp.nextEntry()
        mouse.setPos((0, 0))


# conditions list used to define create trial loops with TrialHandler
conditionsList = CounterBalance(expInfo['participant'], balancingList)

# Initialise credits
credits = 0.0
# Initialise some variables
nReinforcers, nResp = 0, 0
nResp1, nResp2 = 0, 0

# holds key response for trials
mouse_thresh = 0.35
mouse = event.Mouse(win=win)
key_resp = event.BuilderKeyResponse()

### Initialize text, image components used in experiment

# TextStims for messages to participant
continue_msg = visual.TextStim(
    win=win, name='continue_msg',
    text="(Click a trackball button to continue)",
    font='Arial', alignText='center',
    pos=(0, -0.85), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='dimgray', colorSpace='rgb', opacity=1,
    depth=0.0)
countdown_msg = visual.TextStim(
    win=win, name='countdown_msg',
    text="Starting in",
    font='Arial', alignText='center',
    pos=(0, -0.75), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='dimgray', colorSpace='rgb', opacity=1,
    depth=0.0)
answer_in_msg = visual.TextStim(
    win=win, name='countdown_msg',
    text="Answer in",
    font='Arial', alignText='center',
    pos=(0, -0.75), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='dimgray', colorSpace='rgb', opacity=1,
    depth=0.0)
break_msg = visual.TextStim(
    win=win, name='break_msg',
    text="Time for a short break!",
    font='Arial', alignText='center',
    pos=(0, 0.4), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='gray', colorSpace='rgb', opacity=1,
    depth=-1.0)
ITI_msg = visual.TextStim(
    win=win, name='countdown_msg',
    text="The game will resume shortly.\n\n"
         "Please try to remain still. Thank you!",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=0.95 * wh_ratio, ori=0,
    color='gray', colorSpace='rgb', opacity=1,
    depth=-1.0)
wait_msg = visual.TextStim(
    win=win, name='wait_msg',
    text="Waiting for technician to mark ready status...\n\n",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='gray', colorSpace='rgb', opacity=1,
    depth=-1.0)
black_bg = visual.Rect(
    win=win, name='black_bg',
    width=1 * wh_ratio, height=1, units='height', pos=(0, 0), ori=0,
    lineColor=None, fillColor='black', fillColorSpace='rgb', opacity=1,
    depth=0.0)

# TextStims for practice
instr_welcome_txt = visual.TextStim(
    win=win, name='instr_welcome_txt',
    text="Welcome to the experiment!",
    font='Arial', alignText='center',
    pos=(0, 0.3), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_practice_txt = visual.TextStim(
    win=win, name='instr_practice_txt',
    text="Before we begin, let's practice some trackball actions.\n\n"
         "You must roll the trackball all the way from one side to the other.\n\n"
         "Try to keep still while using the trackball.",
    font='Arial', alignText='center',
    pos=(0, -0.1), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

instr_right_txt = visual.TextStim(
    win=win, name='instr_right_txt',
    text="For this figure, you may roll the trackball to the RIGHT.\n\n"
         "Try it a few times!",
    font='Arial', alignText='center',
    pos=(0, 0.4), height=0.1, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_left_txt = visual.TextStim(
    win=win, name='instr_left_txt',
    text="For this figure, you may roll the trackball to the LEFT.\n\n"
         "Try it a few times!",
    font='Arial', alignText='center',
    pos=(0, 0.4), height=0.1, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_buttonleft_txt = visual.TextStim(
    win=win, name='instr_buttonup_txt',
    text="Try collecting the left 2 coins by clicking the left trackball button.",
    font='Arial', alignText='center',
    pos=(0, 0.5), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_buttonright_txt = visual.TextStim(
    win=win, name='instr_buttonright_txt',
    text="Now collect the right 2 coins by clicking the right trackball button.",
    font='Arial', alignText='center',
    pos=(0, 0.5), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

blue1 = visual.ImageStim(
    win=win, name='gold1',
    image='stim' + os.sep + 'blue_coin.png', mask=None,
    ori=0, pos=(-0.3 * wh_ratio, -0.25), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
blue2 = visual.ImageStim(
    win=win, name='gold1',
    image='stim' + os.sep + 'blue_coin.png', mask=None,
    ori=0, pos=(-0.2 * wh_ratio, 0.05), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
blue3 = visual.ImageStim(
    win=win, name='gold1',
    image='stim' + os.sep + 'blue_coin.png', mask=None,
    ori=0, pos=(0.25 * wh_ratio, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
blue4 = visual.ImageStim(
    win=win, name='gold1',
    image='stim' + os.sep + 'blue_coin.png', mask=None,
    ori=0, pos=(0.15 * wh_ratio, -0.3), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)

start_txt = visual.TextStim(
    win=win, name='good_luck_txt',
    text="You are done with practice.\n\n\n\nPress a button when you are ready to begin the task!",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

training_ready_txt = visual.TextStim(
    win=win, name='training_ready_txt',
    text="Use the trackball to make responses.\n\n\nGet ready!",
    font='Arial', alignText='center',
    pos=(0, 0), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
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
    pos=(0, 0), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
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
    depth=0.0)
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
    pos=(0, 0.4), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_consumption = "Use the buttons to collect 10 coins from the left or right of the screen.\n\n" \
                    "You get " + str(fixed_durations['consumption']) + " seconds. Get ready!"
# Initialize components for Routine "instr_before_consumption_test"
instr_consumption_txt = visual.TextStim(
    win=win, name='instr_consumption_txt',
    text=instr_consumption,
    font='Arial', alignText='left',
    pos=(0, -0.1), height=0.1, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "consumption_test"
# Define coin positions for consumption test
left_positions = [(-0.1 * wh_ratio, 0.1),
                  (-0.35 * wh_ratio, 0),
                  (-0.2 * wh_ratio, -0.1),
                  (-0.025 * wh_ratio, -0.1),
                  (-0.25 * wh_ratio, 0.2),
                  (-0.1 * wh_ratio, -0.3),
                  (-0.4 * wh_ratio, 0.3),
                  (-0.4 * wh_ratio, -0.25),
                  (-0.25 * wh_ratio, -0.375),
                  (-0.05 * wh_ratio, 0.35)]
right_positions = [(0.05 * wh_ratio, -0.3),
                   (0.4 * wh_ratio, 0.05),
                   (0.4 * wh_ratio, 0.3),
                   (0.175 * wh_ratio, 0.35),
                   (0.25 * wh_ratio, 0.15),
                   (0.225 * wh_ratio, -0.35),
                   (0.1 * wh_ratio, 0.1),
                   (0.15 * wh_ratio, -0.15),
                   (0.3 * wh_ratio, -0.1),
                   (0.4 * wh_ratio, -0.3)]
shuffle(left_positions)
shuffle(right_positions)

# Initialize 10 gold and 10 silver coins
gold1 = visual.ImageStim(
    win=win, name='gold1',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
gold2 = visual.ImageStim(
    win=win, name='gold2',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
gold3 = visual.ImageStim(
    win=win, name='gold3',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
gold4 = visual.ImageStim(
    win=win, name='gold4',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
gold5 = visual.ImageStim(
    win=win, name='gold5',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
gold6 = visual.ImageStim(
    win=win, name='gold6',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-6.0)
gold7 = visual.ImageStim(
    win=win, name='gold7',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-7.0)
gold8 = visual.ImageStim(
    win=win, name='gold8',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-8.0)
gold9 = visual.ImageStim(
    win=win, name='gold9',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-9.0)
gold10 = visual.ImageStim(
    win=win, name='gold10',
    image='stim' + os.sep + 'gold_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-10.0)
silver1 = visual.ImageStim(
    win=win, name='silver1',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-11.0)
silver2 = visual.ImageStim(
    win=win, name='silver2',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-12.0)
silver3 = visual.ImageStim(
    win=win, name='silver3',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-13.0)
silver4 = visual.ImageStim(
    win=win, name='silver4',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-14.0)
silver5 = visual.ImageStim(
    win=win, name='silver5',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-15.0)
silver6 = visual.ImageStim(
    win=win, name='silver6',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-16.0)
silver7 = visual.ImageStim(
    win=win, name='silver7',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-17.0)
silver8 = visual.ImageStim(
    win=win, name='silver8',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-18.0)
silver9 = visual.ImageStim(
    win=win, name='silver9',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-19.0)
silver10 = visual.ImageStim(
    win=win, name='silver10',
    image='stim' + os.sep + 'silver_coin.png', mask=None,
    ori=0, pos=(0, 0), size=(0.15, 0.15), units='height',
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

curtain_txt = visual.TextStim(
    win=win, name='curtain_txt',
    text="CURTAIN",
    font='Arial', alignText='center',
    pos=(0, 0.55), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr_choice_test_str = "Use the trackball to make responses.\n\n" \
                        "The coin and response red triangle will not be visible. Nothing else about the game has changed.\n\n" \
                        "You will get " + str(fixed_durations['choice_test']) + " seconds.  Ready?"
# Initialize components for Routine "instr_choice_test"
instr_choice_test_txt = visual.TextStim(
    win=win, name='instr_choice_test_txt',
    text=instr_choice_test_str,
    font='Arial', alignText='left',
    pos=(0, -0.15), height=0.1, wrapWidth=0.95 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "choice_test"
stim_choice1 = visual.ImageStim(
    win=win, name='stim_choice1',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
stim_choice2 = visual.ImageStim(
    win=win, name='stim_choice_2',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
curtain_img = visual.ImageStim(
    win=win, name='curtain_img',
    image='stim' + os.sep + 'curtain.png', mask=None,
    ori=0, pos=(0, 0), size=(0.2, 0.2), units='height',
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
    depth=0.0)

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
    pos=(0, 0.4), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
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
    pos=(0, 0.4), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-3.0)

# Initialize components for Routine "goodbye"
final_txt = visual.TextStim(
    win=win, name='final_txt',
    text="End of the experiment.\n\nThanks for participating!",
    font='Arial', alignText='center',
    pos=(0, 0.2), height=0.1, wrapWidth=0.9 * wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
credits_txt = visual.TextStim(
    win=win, name='credits_txt',
    text='default text',
    font='Arial', alignText='center',
    pos=(0, -0.35), height=0.15, wrapWidth=0.9 * wh_ratio, ori=0,
    color='yellow', colorSpace='rgb', opacity=1,
    depth=-1.0)

if not debugging:
    win.mouseVisible = False

####### BEGIN PRACTICE  #########
show_instr([instr_welcome_txt, instr_practice_txt], continue_msg, routineClock, win, mouse, 2)

# set up handler to look after randomisation of conditions etc
practice_trialLoop = data.TrialHandler(nReps=1, method='sequential',
                                       extraInfo=expInfo, originPath=-1,
                                       trialList=conditionsList[:2],
                                       seed=None, name='practice_trialLoop')
for thisTrial in practice_trialLoop:
    currentLoop = practice_trialLoop
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    phase = 'practice'

    # ------Prepare to start Routine "practice"-------
    continueRoutine = True
    # update component parameters for each repeat
    stim.setPos(position)
    stim.setImage(stimulus)

    allowed_resp_txt = instr_right_txt if corrResp == "right" else instr_left_txt
    practiceComponents = [stim, allowed_resp_txt, triangle_resp]
    nResp = 0  # reset response counter
    corr = False

    set_components_attr(practiceComponents + [continue_msg], 'status', NOT_STARTED)

    prev_buttons = mouse.getPressed()
    mouse.setPos((0, 0))
    # -------Start Routine "practice"-------
    while continueRoutine:
        theseKeys = event.getKeys(keyList=['escape', 'q'])
        # check for quit:
        if theseKeys:
            core.quit()

        # update/draw components on each frame
        for component in practiceComponents:
            if component.status == NOT_STARTED:
                component.setAutoDraw(True)
                event.clearEvents(eventType='keyboard')
        if triangle_resp.status == STARTED:  # only update if drawing
            triangle_resp.setOpacity(0, log=False)

        (x, y) = mouse.getPos()
        if x > mouse_thresh:
            resp_made('right')
        if x < -mouse_thresh:
            resp_made('left')

        # allow participant to continue after practicing response 3 times
        if nResp >= 3:
            if continue_msg.status == NOT_STARTED:
                continue_msg.setAutoDraw(True)
            if continue_msg.status == STARTED:
                buttons = mouse.getPressed()
                if all([item == 0 for item in prev_buttons]):
                    if any(buttons):
                        # a response ends the routine
                        continueRoutine = False
                prev_buttons = buttons

                if not continueRoutine:
                    continue_msg.setAutoDraw(False)

        if not continueRoutine:  # end routine
            set_components_attr(practiceComponents, 'autoDraw', False)
        # refresh the screen
        win.flip()

    # after practicing both trackball responses, practice button responses
    if practice_trialLoop.thisN == 1:
        continueRoutine = True

        practice_coins = [blue1, blue2, blue3, blue4]
        # topComponents = [, gold1, silver1]
        # bottomComponents = [, gold2, silver2]
        set_components_attr(practice_coins + [instr_buttonleft_txt, instr_buttonright_txt], 'status', NOT_STARTED)
        set_components_attr(practice_coins + [instr_buttonleft_txt], 'autoDraw', True)

        prev_buttons = mouse.getPressed()
        while continueRoutine:
            theseKeys = event.getKeys(keyList=['escape', 'q'])
            # check for quit:
            if theseKeys:
                core.quit()

            buttons = mouse.getPressed()
            if all([item == 0 for item in prev_buttons]):
                # check for top button clicks
                if buttons[0] and not buttons[2]:
                    if len(practice_coins) > 2:
                        pop_coin = practice_coins.pop(0)
                        pop_coin.setAutoDraw(False)
                    # the top two coins have been collected
                    if len(practice_coins) == 2 and instr_buttonright_txt.status == NOT_STARTED:
                        instr_buttonleft_txt.setAutoDraw(False)
                        instr_buttonright_txt.setAutoDraw(True)
                if buttons[2] and not buttons[0]:
                    if 3 > len(practice_coins) > 0:
                        pop_coin = practice_coins.pop(0)
                        pop_coin.setAutoDraw(False)
                    if len(practice_coins) == 0:  # all coins have been collected
                        instr_buttonright_txt.setAutoDraw(False)
                        continueRoutine = False
            prev_buttons = buttons

            # keep mouse in this window
            (x, y) = mouse.getPos()
            if x > 0.5:
                mouse.setPos((0.5, y))
            if x < -0.5:
                mouse.setPos((-0.5, y))

            win.flip()

# participant clicks button when they are ready to begin
show_instr([start_txt], None, routineClock, win, mouse, 0)

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
            if blockLoop.thisTrialN == 0:
                MR_settings['volumes'] = run1_volumes
            else:
                MR_settings['volumes'] = run2_volumes
            # wait for spacebar to mark ready for scanner
            wait_msg.draw()
            win.flip()
            event.clearEvents(eventType='keyboard')
            theseKeys = event.waitKeys(keyList=['escape', 'q', 'space'])
            if 'escape' in theseKeys or 'q' in theseKeys:
                core.quit()
            elif 'space' in theseKeys:
                win.flip()
            # Ready to start run! Wait for scanner pulse with launchScan(), which will reset the global clock
            pulse = launchScan(win, MR_settings, globalClock=globalClock, mode=emulator_mode)

            reminder_duration = fixed_durations['training_ready']
            trialLoop.addData("run", blockLoop.thisTrialN)
            trialLoop.addData("event", "instr_before_block")
            trialLoop.addData("duration", reminder_duration)
            trialLoop.addData("globalClock_t", str(globalClock.getTime()))
            thisExp.nextEntry()
            routineTimer.reset(reminder_duration)
            show_timed_countdown([training_ready_txt], countdown_msg, routineTimer, win, mouse, countdown_t_minus=5)

        currentLoop = trialLoop
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))

        trial_duration = trial_durations[blockLoop.thisTrialN][trialLoop.thisTrialN][trialLoop.thisRepN]
        print("trial dur: " + str(trial_duration))

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
        print("nResp until next rnf: " + str(nResp_until_next_rnf))

        # add entry for training start
        trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        trialLoop.addData("routineClock_t", str(t))
        trialLoop.addData("run", blockLoop.thisTrialN)
        trialLoop.addData("trial", nBlocks * nTrials * blockLoop.thisN + trialLoop.thisN)
        trialLoop.addData("event", "training_period_start")
        trialLoop.addData("duration", trial_duration)
        trialLoop.addData("nResp_untilRnf", nResp_until_next_rnf)
        thisExp.nextEntry()

        coin.setImage(coin_img)
        # keep track of which components have finished
        trainingComponents = [triangle_resp, stim, coin]
        set_components_attr(trainingComponents, 'status', NOT_STARTED)
        mouse.setPos((0, 0))

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

            framesRemain = 0.0 + trial_duration - 20 * win.monitorFramePeriod
            if t <= framesRemain:
                (x, y) = mouse.getPos()
                if x > mouse_thresh:
                    resp_made('right')
                if x < -mouse_thresh:
                    resp_made('left')
            else:
                continueRoutine = False

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                set_components_attr(trainingComponents, 'autoDraw', False)
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        # -------Ending Routine "training"-------

        # ------Prepare to start Routine "ITI"-------
        ITI_duration = ITI_durations[blockLoop.thisTrialN][
            len(trialLoop.trialList) * trialLoop.thisRepN + trialLoop.thisTrialN]
        print("ITI: " + str(ITI_duration) + "\n")
        routineTimer.reset(ITI_duration)
        # add entry for ITI start
        trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        trialLoop.addData("routineClock_t", str(t))
        trialLoop.addData("run", blockLoop.thisTrialN)
        trialLoop.addData("event", "ITI_start")
        trialLoop.addData("duration", ITI_duration)
        thisExp.nextEntry()
        # ------Starting Routine "ITI"-------
        # show countdown if there is another trial coming up after this ITI
        if trialLoop.thisRepN == nTrials - 1 and trialLoop.thisTrialN == len(trialLoop.trialList) - 1:
            show_timed_countdown([ITI_msg, break_msg], None, routineTimer, win, mouse, countdown_t_minus=5, bg_rect=black_bg)
        else:
            show_timed_countdown([ITI_msg, break_msg], countdown_msg, routineTimer, win, mouse, countdown_t_minus=5,
                                 bg_rect=black_bg)
        # -------Ending Routine "ITI"-------

    if blockLoop.thisTrialN == 0:
        # ------ Start Routine "piggy_bank_partially_full"-------
        piggy_msg_duration = fixed_durations['piggy_bank_partially_full']
        routineTimer.reset(piggy_msg_duration)
        # add entry for piggy message start
        thisExp.addData("globalClock_t", str(globalClock.getTime()))
        thisExp.addData("event", "piggy_getting_full_msg")
        thisExp.addData("run", blockLoop.thisTrialN)
        thisExp.addData("duration", piggy_msg_duration)
        thisExp.nextEntry()
        piggy_bank_partially_fullComponents = [piggy_part_txt]
        show_timed_countdown(piggy_bank_partially_fullComponents, None, routineTimer, win, mouse)
        # ------ End Routine "piggy_bank_partially_full"-------
    else:
        # ------Start Routine "piggy_bank_full"-------
        devalued_piggy_img.setImage(silver_piggy_fname if devalued_coin == "silver" else gold_piggy_fname)
        piggy_msg_duration = fixed_durations['piggy_bank_full']
        routineTimer.reset(piggy_msg_duration)
        # add entry for piggy message start
        thisExp.addData("globalClock_t", str(globalClock.getTime()))
        thisExp.addData("event", "piggy_full_msg")
        thisExp.addData("run", blockLoop.thisTrialN)
        thisExp.addData("duration", piggy_msg_duration)
        thisExp.nextEntry()
        piggy_bank_fullComponents = [piggy_full_txt, devalued_piggy_img]
        show_timed_countdown(piggy_bank_fullComponents, None, routineTimer, win, mouse)
        # ------End Routine "piggy_bank_full"-------

    # ------Prepare to start Routine "instr_before_consumption_test"-------
    consumption_before_duration = fixed_durations['instr_before_consumption_test']
    routineTimer.reset(consumption_before_duration)
    thisExp.addData("globalClock_t", globalClock.getTime())
    thisExp.addData("event", 'instr_before_consumption')
    thisExp.addData("run", blockLoop.thisTrialN)
    thisExp.addData("duration", consumption_before_duration)
    thisExp.nextEntry()
    instr_before_consumption_testComponents = [free_collection_txt, instr_consumption_txt]
    show_timed_countdown(instr_before_consumption_testComponents, countdown_msg, routineTimer, win, mouse, countdown_t_minus=5)

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

    right_cointype = outcome_choice1
    # keep track of which components have finished
    if right_cointype == 'gold':
        right_coins = gold_coins
        left_coins = silver_coins
    else:
        right_coins = silver_coins
        left_coins = gold_coins
    delay_start_t = 0
    prev_buttons = mouse.getPressed()

    consumption_duration = fixed_durations['consumption']
    blockLoop.addData("globalClock_t", globalClock.getTime())
    blockLoop.addData("routineClock_t", str(t))
    blockLoop.addData("event", 'consumption_start')
    blockLoop.addData("run", blockLoop.thisTrialN)
    blockLoop.addData("duration", consumption_duration)
    thisExp.nextEntry()

    # routineTimer.reset(consumption_duration)
    consumption_testComponents = gold_coins + silver_coins
    set_components_attr(consumption_testComponents, 'status', NOT_STARTED)

    # -------Start Routine "consumption_test"-------
    while continueRoutine:
        # get current time
        # t_remain = routineTimer.getTime()
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

        frameRemains = 0.0 + consumption_duration - win.monitorFramePeriod * 0.75  # most of one frame period left
        if t < frameRemains:

            for idx in range(len(left_coins)):
                component = left_coins[idx]
                if t < frameRemains:
                    if component.status == NOT_STARTED:
                        component.setPos(left_positions[idx])
                        component.setOpacity(1, log=False)
                        component.setAutoDraw(True)
                        event.clearEvents(eventType='keyboard')
                else:
                    if component.status == STARTED:
                        component.setAutoDraw(False)

            for idx in range(len(right_coins)):
                component = right_coins[idx]
                if t < frameRemains:
                    if component.status == NOT_STARTED:
                        component.tStart = t
                        component.frameNStart = frameN  # exact frame index
                        component.setPos(right_positions[idx])
                        component.setOpacity(1, log=False)
                        component.setAutoDraw(True)
                else:
                    if component.status == STARTED:
                        component.setAutoDraw(False)

            if len(selected_coins) < 10:
                buttons = mouse.getPressed()
                if all([item == 0 for item in prev_buttons]):
                    left_cnt = buttons[0]
                    right_cnt = buttons[2]
                    if left_cnt > right_cnt == 0:
                        if left_coins:
                            selected = left_coins.pop()
                            selected_coins.append(selected)
                            selected.setAutoDraw(False)
                            resp_made('left')
                    elif right_cnt > left_cnt == 0:
                        if right_coins:
                            selected = right_coins.pop()
                            selected_coins.append(selected)
                            selected.setAutoDraw(False)
                            resp_made('right')
                prev_buttons = buttons
        else:
            set_components_attr(consumption_testComponents, 'autoDraw', False)
            key_resp.status = FINISHED
            break

        if len(selected_coins) == 10:
            if not delay_start_t:
                delay_start_t = t
                left_coins = []
                right_coins = []
            else:
                if t > delay_start_t + 0.1:
                    for component in consumption_testComponents:
                        if component.status == STARTED:
                            if hasattr(component, 'draw'):
                                component.setOpacity(0.2, log=False)

        # keep mouse in this window
        (x, y) = mouse.getPos()
        if x > 0.9:
            mouse.setPos((0.9, y))
        if x < -0.9:
            mouse.setPos((-0.9, y))

        # refresh the screen
        win.flip()
    # -------Ending Routine "consumption_test"-------

    # ------Prepare to start Routine "instr_after_consumption_test"-------
    consumption_after_duration = fixed_durations['instr_after_consumption_test']
    blockLoop.addData("globalClock_t", str(globalClock.getTime()))
    blockLoop.addData("routineClock_t", str(t))
    blockLoop.addData("event", "instr_after_consumption")
    blockLoop.addData("run", blockLoop.thisTrialN)
    blockLoop.addData("duration", consumption_after_duration)
    thisExp.nextEntry()
    # keep track of which components have finished
    routineTimer.reset(consumption_after_duration)  # clock
    instr_after_consumption_testComponents = [instr_after_consumption_test]
    show_timed_countdown(instr_after_consumption_testComponents, None, routineTimer, win, mouse)
    # -------Ending Routine "instr_after_consumption_test"-------

    # ------Prepare to start Routine "rest_after_block"-------'
    after_block_duration = fixed_durations['after_block']
    thisExp.addData("globalClock_t", str(globalClock.getTime()))
    thisExp.addData("event", "rest_after_block")
    thisExp.addData("run", blockLoop.thisTrialN)
    thisExp.addData("duration", after_block_duration)
    thisExp.nextEntry()
    routineTimer.reset(after_block_duration)  # clock
    show_timed_countdown([], None, routineTimer, win, mouse)
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

    # ------Prepare to start Routine "instr_choice_test"-------
    instr_choice_test_duration = fixed_durations['instr_choice_test']
    choice_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    choice_trialLoop.addData("event", "instr_before_choice_test")
    choice_trialLoop.addData("run", blockLoop.nTotal-1)
    choice_trialLoop.addData("duration", instr_choice_test_duration)
    thisExp.nextEntry()
    routineTimer.reset(instr_choice_test_duration)  # clock
    show_timed_countdown([curtain_txt, instr_choice_test_txt], countdown_msg, routineTimer, win, mouse, countdown_t_minus=5)

    # ------Prepare to start Routine "choice_test"-------
    t = 0
    routineClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    stim_choice1.setPos(position1)
    stim_choice1.setImage(stimulus_choice1)
    stim_choice2.setPos(position2)
    stim_choice2.setImage(stimulus_choice2)

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
    print("nResp until next rnf: " + str(nResp_until_next_rnf))
    choice_duration = fixed_durations['choice_test']
    choice_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    choice_trialLoop.addData("routineClock_t", str(t))
    choice_trialLoop.addData("event", 'choice_test_start')
    choice_trialLoop.addData("run", blockLoop.nTotal-1)
    choice_trialLoop.addData("duration", choice_duration)
    choice_trialLoop.addData("nResp_untilRnf", nResp_until_next_rnf)
    thisExp.nextEntry()

    # keep track of which components have finished
    choice_testComponents = [triangle_resp, stim_choice1, stim_choice2, curtain_img]
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
            resp_made('right')
        if x < -mouse_thresh:
            resp_made('left')

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
        contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
        contingency_trialLoop.addData("event", "instr_before_contingency_test")
        contingency_trialLoop.addData("duration", instr_contingency_test_duration)
        thisExp.nextEntry()
        routineTimer.reset(instr_contingency_test_duration)  # clock
        # show_timed([], routineTimer, win, static_period)
        instr_contingency_testComponents = [instr_contingency_test_txt]
        show_timed_countdown(instr_contingency_testComponents, countdown_msg, routineTimer, win, mouse, countdown_t_minus=5)

    # ------Prepare to start Routine "contingency_test"-------

    # short pause before each question
    pause_duration = fixed_durations["pause_between_qs"]
    routineTimer.reset(pause_duration)
    contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    contingency_trialLoop.addData("event", "pause_before_contingency_question")
    contingency_trialLoop.addData("duration", pause_duration)
    thisExp.nextEntry()
    show_timed_countdown([], None, routineTimer, win, mouse)

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
        contingency_trialLoop.addData("event", "contingency_start_coin")
    else:
        stim.setImage(stimulus)
        stim.setPos(position)
        contingency_test_img = stim
        contingency_test_txt = stim_contingency_txt
        contingency_trialLoop.addData("event", "contingency_start_stim")

    contingency_testComponents = [contingency_test_img, contingency_test_txt]
    nResp = 0
    contingency_trialLoop.addData("globalClock_t", str(globalClock.getTime()))
    contingency_trialLoop.addData("routineClock_t", str(t))
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
            resp_made('right')
            continueRoutine = False
        if x < -mouse_thresh:
            resp_made('left')
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
thisExp.addData("event", "goodbye_msg_start")
thisExp.addData("duration", goodbye_duration)
thisExp.nextEntry()

goodbyeComponents = [final_txt, credits_txt]
routineTimer.reset(goodbye_duration)
show_timed_countdown(goodbyeComponents, None, routineTimer, win, mouse)

thisExp.addData("globalClock_t", str(globalClock.getTime()))
# thisExp.addData("routineClock_t", str(t))
thisExp.addData("event", "goodbye_msg_end")

# compute and display payout for the session
payout_this_session = credits / 100
print("payout: " + str(payout_this_session))

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename + '.csv')
thisExp.saveAsPickle(filename)

# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
