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


# Ensure that relative paths start from the same directory as this script
_thisDir = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(str(_thisDir))

mouse_thresh = 0.5

logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation
# fullscreen = not debugging
debugging = True
fullscreen = True
size = [1920, 1080] if not debugging else [800, 600]
# Setup the Window
win = visual.Window(
    size=size, fullscr=fullscreen, screen=1,
    allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0.506, 0.506, 0.506], colorSpace='rgb',
    blendMode='avg', useFBO=True)
wh_ratio = win.size[0]/win.size[1]

expInfo = {}
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# holds key response for trials
mouse_thresh = 0.5
mouse=event.Mouse(win=win)
key_resp = event.BuilderKeyResponse()

### Initialize text, image components used in experiment
instrClock = core.Clock()

continue_msg = visual.TextStim(
    win=win, name='continue_msg',
    text="(Click anywhere to continue)",
    font='Arial', alignText='center',
    pos=(0, -0.75), height=0.06, wrapWidth=None, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)


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
    pos=(0,-0.1), height=0.06, wrapWidth=0.75*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-1.0)
show_instr([instr0a_txt, instr0b_txt], continue_msg, instrClock, win, mouse, 0)

instr1 = "In this experiment, you will have the chance to collect some coins.\n\n\n\n\n\n\n\n" \
         "Each coin collected is worth 40 CREDITS.\n" \
         "Gold and silver coins have the same value."
instr1_txt = visual.TextStim(
    win=win, name='instr1_txt',
    text=instr1,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.45*wh_ratio, ori=0,
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
show_instr([instr1_txt, silver_coin_img, gold_coin_img], continue_msg, instrClock, win, mouse, 0)


instr2 = "The number of credits you have at the end of the experiment will be used to calculate a monetary bonus.\n\n" \
         "The bonus will be added to your payment for this experiment."
instr2_txt = visual.TextStim(
    win=win, name='instr2_txt',
    text=instr2,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.65*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr2_txt], continue_msg, instrClock, win, mouse, 0)



instr3 = "During the task, you will be allowed to perform one of two actions with the trackball:\n" \
         "    -   roll left, or \n" \
         "    -   roll right\n\n" \
         "You will know which action is allowed by a figure that will appear on the screen.\n\n" \
         "Each allowed action has a cost of 1 CREDIT."
instr_3_txt = visual.TextStim(
    win=win, name='instr_3_txt',
    text=instr3,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.85*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr_3_txt], continue_msg, instrClock, win, mouse, 0)


instr4a = "When you see this figure, you will be allowed to respond by rolling the trackball to the right.\n\n" \
         "Try moving the mouse right to mimic this action."
instr_4a_txt = visual.TextStim(
    win=win, name='instr_4a_txt',
    text=instr4a,
    font='Arial', alignText='left',
    pos=(0, 0.45), height=0.06, wrapWidth=0.5*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr4b = "Every time you perform the correct action for the figure, a triangle will appear at the top-right of the screen signalling that a response has been made.\n\n\n" \
         "(Keep practicing, or click anywhere to continue)"
instr_4b_txt = visual.TextStim(
    win=win, name='instr_4b_txt',
    text=instr4b,
    font='Arial', alignText='left',
    pos=(0, -0.3), height=0.06, wrapWidth=0.5*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=-1.0);
stim1_img = visual.ImageStim(
    win=win, name='stim1_img',
    image='stim' + os.sep + 'right_resp_img.png', mask=None,
    ori=0, pos=(-0.5, 0), size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
triangle_resp = visual.ShapeStim(  # also used for Routine "choice_test"
    win=win, name='triangle_resp',
    vertices=[[-(0.2, 0.2)[0] / 2.0, -(0.2, 0.2)[1] / 2.0], [+(0.2, 0.2)[0] / 2.0, -(0.2, 0.2)[1] / 2.0],
              [0, (0.2, 0.2)[1] / 2.0]],
    ori=0, pos=(0.85, 0.85),
    lineWidth=2, lineColor=[1.000, 0.004, -1.000], lineColorSpace='rgb',
    fillColor=[1.000, -1.000, -1.000], fillColorSpace='rgb',
    opacity=1.0, depth=-3.0, interpolate=True)

# ------Prepare to start Routine "resp1"-------
frameN = -1
continueRoutine = True
# update component parameters for each repeat
resp1_practComponents = [instr_4a_txt, stim1_img, triangle_resp]
instr_4b_txt.status = NOT_STARTED
set_components_attr(resp1_practComponents, 'autoDraw', True)
mouse.setPos((0, 0))
prev_buttons = mouse.getPressed()
mouse_allowed = False
while continueRoutine:
    # get current time
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    theseKeys = event.getKeys(keyList=['escape', 'q'])
    # check for quit:
    if theseKeys:
        core.quit()

    triangle_resp.setOpacity(0, log=False)
    (x, y) = mouse.getPos()
    if x > mouse_thresh:
        for frameN in range(5):  # show for bit more than half a sec (60 frames = 1 sec)
            triangle_resp.setOpacity(1)
            win.flip()
        mouse.setPos((0, 0))
        if instr_4b_txt.status == NOT_STARTED:
            instr_4b_txt.setAutoDraw('True')
            resp1_practComponents.append(instr_4b_txt)
        mouse_allowed = True
        # continueRoutine = False
    # check if ready to continue
    if mouse_allowed:
        buttons = mouse.getPressed()
        if all([item==0 for item in prev_buttons]):
            if any(buttons):
                # a response ends the routine
                set_components_attr(resp1_practComponents, 'autoDraw', False)
                continueRoutine = False
        prev_buttons = buttons
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    # refresh the screen
    win.flip()



instr5a = "When you see this figure, you will be allowed to respond by rolling the trackball to the left.\n\n" \
         "Try moving the mouse left to mimic this action."
instr_5a_txt = visual.TextStim(
    win=win, name='instr_5a_txt',
    text=instr5a,
    font='Arial', alignText='left',
    pos=(0, 0.45), height=0.06, wrapWidth=0.5*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
instr5b = "(Keep practicing, or click anywhere to continue)"
instr_5b_txt = visual.TextStim(
    win=win, name='instr_5_txt',
    text=instr5b,
    font='Arial', alignText='left',
    pos=(0, -0.1), height=0.06, wrapWidth=0.5*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0);
stim2_img = visual.ImageStim(
    win=win, name='stim2_img',
    image='stim' + os.sep + 'left_resp_img.png', mask=None,
    ori=0, pos=(0.5, 0), size=(0.25, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
# ------Prepare to start Routine "resp2"-------
t = 0
instrClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
resp2_practComponents = [instr_5a_txt, stim2_img, triangle_resp]
instr_5b_txt.status = NOT_STARTED
set_components_attr(resp2_practComponents, 'autoDraw', True)
mouse.setPos((0, 0))
prev_buttons = mouse.getPressed()
mouse_allowed = False
# -------Start Routine "contingency_test"-------
while continueRoutine:
    # get current time
    t = instrClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)

    theseKeys = event.getKeys(keyList=['escape', 'q'])
    # check for quit:
    if "escape" in theseKeys or 'q' in theseKeys:
        endExpNow = True
        continueRoutine = False

    triangle_resp.setOpacity(0, log=False)
    (x, y) = mouse.getPos()
    if x < -mouse_thresh:
        for frameN in range(5):  # show for bit more than half a sec (60 frames = 1 sec)
            triangle_resp.setOpacity(1)
            win.flip()
        mouse.setPos((0, 0))
        if instr_5b_txt.status == NOT_STARTED:
            instr_5b_txt.setAutoDraw('True')
            resp2_practComponents.append(instr_5b_txt)
        mouse_allowed = True
        # continueRoutine = False
    # check if all components have finished
    if mouse_allowed:
        buttons = mouse.getPressed()
        if all([item==0 for item in prev_buttons]):
            if any(buttons):
                # a response ends the routine
                set_components_attr(resp2_practComponents, 'autoDraw', False)
                continueRoutine = False
        prev_buttons = buttons
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape", 'q']):
        core.quit()
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    # refresh the screen
    win.flip()


instr6 = "Your trackball actions will *sometimes* be rewarded with a coin (either silver or gold).\n\n" \
         "IMPORTANT: One of the actions (eg., left) will give you gold coins when you are rewarded; the other action (eg., right) will give you silver coins when you are rewarded.\n" \
         "This will remain the same throughout the whole experiment."
instr_6_txt = visual.TextStim(
    win=win, name='instr_6_txt',
    text=instr6,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.65*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr_6_txt], continue_msg, instrClock, win, mouse, 0)


instr7 = "The coin (either silver or gold) at the center of the screen will light up every time you are rewarded for an action."
instr_7_txt = visual.TextStim(
    win=win, name='instr_7_txt',
    text=instr7,
    font='Arial', alignText='left',
    pos=(0, 0.4), height=0.06, wrapWidth=0.6*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
light_up_coins_img = visual.ImageStim(
    win=win, name='light_up_coins_img',
    image='stim' + os.sep + 'coins_light_up_instructions.png', mask=None,
    ori=0, pos=(0, -0.05), size=(0.75, 0.4), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
show_instr([instr_7_txt, light_up_coins_img], continue_msg, instrClock, win, mouse, 0)


instr8 = "There are 2 piggy banks:\n\n\n\n\n\n\n\n\n" \
         "Rewarded coins are stored in their corresponding piggy banks, provided they are not full.\n" \
         "A coin deposited into a piggy bank earns you 40 CREDITS.\n\n" \
         "You will be notified if a piggy bank becomes full."
instr_8_txt = visual.TextStim(
    win=win, name='instr_8_txt',
    text=instr8,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.65*wh_ratio, ori=0,
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
show_instr([instr_8_txt, silver_piggy_img, gold_piggy_img], continue_msg, instrClock, win, mouse, 0)


instr9 = "Given the nature of the task, not every correct trackball action you make will reward you with a coin.\n\n" \
         "It is your task to discover the best way of performing the actions to earn as many credits as possible during the experiment.\n\n" \
         "The best strategy will be the same for both actions, since they only differ in whether they give you silver or gold coins."
instr_9_txt = visual.TextStim(
    win=win, name='instr_9_txt',
    text=instr9,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.75*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr_9_txt], continue_msg, instrClock, win, mouse, 0)

instr10 = "During the FREE COLLECTION phase of the experiment, you will be given 15 seconds to press the buttons beside the trackball to select up to 10 coins from the screen.\n\n" \
         "The left button will select coins on the left side of the screen, and the right button will select coins on the right side of the screen.\n" \
         "You will have a chance to practice these actions once you are in the scanner.\n\n" \
         "Coins you pick up in the FREE COLLECTION phase will also be deposited into their respective piggy banks for 40 CREDITS each (provided the piggy bank is not full)."
instr_10_txt = visual.TextStim(
    win=win, name='instr_10_txt',
    text=instr10,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.9*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr_10_txt], continue_msg, instrClock, win, mouse, 0)

instr11 = "During the CURTAIN phase of the experiment, you will be given 90 seconds to again make responses by rolling the trackball.\n\n" \
          "This time, the coin at the center of the screen will be covered with this curtain:\n\n\n\n\n\n\n\n" \
          "and you will *not* see the red triangle when you make a response.\n\n" \
          "Apart from this, nothing else about the game has changed. \n" \
          "Allowed actions will still be denoted by their corresponding figures, " \
          "each action will still be associated with the same type of coin, " \
          "and awarded coins will be deposited in their respective piggy banks for 40 CREDITS (provided the piggy bank is not full)."
curtain_img = visual.ImageStim(
    win=win, name='curtain_img',
    image='stim' + os.sep + 'curtain.png', mask=None,
    ori=0, pos=(0, 0.03), size=(0.2, 0.2), units='height',
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
instr_11_txt = visual.TextStim(
    win=win, name='instr_11_txt',
    text=instr11,
    font='Arial', alignText='left',
    pos=(0, 0), height=0.06, wrapWidth=0.9*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr_11_txt, curtain_img], continue_msg, instrClock, win, mouse, 0)


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
    pos=(0, 0), height=0.06, wrapWidth=0.9*wh_ratio, ori=0,
    color='black', colorSpace='rgb', opacity=1,
    depth=0.0)
show_instr([instr_12_txt], None, instrClock, win, mouse, 0)

# make sure everything is closed down
win.close()
core.quit()
