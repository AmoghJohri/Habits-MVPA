#!/usr/bin/env python
# -*- coding: utf-8 -*-
from psychopy import core, event
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
from math import ceil, floor

def set_components_attr(components_list, attr_name, attr_val):
    for thisComponent in components_list:
        if hasattr(thisComponent, attr_name):
            setattr(thisComponent, attr_name, attr_val)


def show_instr(components_list, continue_msg, clock, win, mouse, t_allow_continue, listen_keys=None):
    clock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # keep track of which components have finished
    set_components_attr(components_list, 'status', NOT_STARTED)
    if continue_msg:
        set_components_attr([continue_msg], 'status', NOT_STARTED)

    prev_buttons = mouse.getPressed()

    if not listen_keys:
        listen_keys = []

    while continueRoutine:
        t = clock.getTime()

        for component in components_list:
            if not isinstance(component, event.BuilderKeyResponse):
                # start all components but key response listener at t=0
                if t >= 0 and component.status == NOT_STARTED:
                    component.setAutoDraw(True)
            else:
                # start key resp listener at t=t_allow_continue
                if t >= t_allow_continue and component.status == NOT_STARTED:
                    component.status = STARTED
                if component.status == STARTED:
                    if listen_keys:
                        theseKeys = event.getKeys(listen_keys)
                        # check for continue key
                        if any([item in theseKeys for item in listen_keys]):
                            continueRoutine = False
            if t >= t_allow_continue:
                buttons = mouse.getPressed()
                if all([item==0 for item in prev_buttons]):
                    if any(buttons):
                        # a response ends the routine
                        continueRoutine = False
                prev_buttons = buttons
                if continue_msg:
                    if continue_msg.status == NOT_STARTED:
                        continue_msg.setAutoDraw(True)
                    if continue_msg.status == STARTED:
                        if not continueRoutine:
                            continue_msg.setAutoDraw(False)

        # check for quit
        theseKeys = event.getKeys(keyList=['escape', 'q'])
        if theseKeys:
            core.quit()

        if not continueRoutine:  # end routine
            set_components_attr(components_list, 'autoDraw', False)
            set_components_attr(components_list, 'status', FINISHED)

        # keep mouse in this window
        (x, y) = mouse.getPos()
        if x > 0.9:
            mouse.setPos((0.9, y))
        if x < -0.9:
            mouse.setPos((-0.9, y))

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()


def show_timed_countdown(components_list, countdown_component, timer, win, mouse, countdown_t_minus=0, listen_keys=None, bg_rect=None):
    frameN = -1
    continueRoutine = True
    # keep track of which components have finished
    set_components_attr(components_list, 'status', NOT_STARTED)
    if countdown_component:
        countdown_component.status = NOT_STARTED
    if bg_rect:
        bg_rect.status = NOT_STARTED
        bg_rect.setOpacity(1)
    status = NOT_STARTED

    countdown_t_minus = int(min(timer.getTime(), countdown_t_minus))
    # -------Start Routine -------
    while continueRoutine:
        # get current time
        t_remain = timer.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *instr_stim1_txt* updates
        frameRemains = win.monitorFramePeriod*0.75
        if t_remain >= frameRemains:
            if status == NOT_STARTED:
                if bg_rect:
                    bg_rect.setAutoDraw(True)
                for component in components_list:
                    component.setAutoDraw(True)
                status = STARTED
                event.clearEvents(eventType='keyboard')
        else:
            if status == STARTED:
                if bg_rect:
                    bg_rect.setAutoDraw(False)
                for component in components_list:
                    component.setAutoDraw(False)
                status = FINISHED

        if countdown_component:
            if t_remain <= countdown_t_minus and countdown_component.status == NOT_STARTED:
                if bg_rect:
                    bg_rect.setOpacity(0)
                countdown_text = countdown_component.text.split(':')[0] + ": " + str(countdown_t_minus)
                countdown_component.setText(countdown_text)
                countdown_component.setAutoDraw(True)
                prev_countdown_val = countdown_t_minus

            if countdown_component.status == STARTED:
                if t_remain >= frameRemains:
                    countdown_val = ceil(t_remain)
                    if countdown_val < prev_countdown_val:
                        countdown_text = countdown_component.text.split(':')[0] + ": " + str(countdown_val)
                        countdown_component.setText(countdown_text)
                        prev_countdown_val = countdown_val
                else:
                    countdown_component.setAutoDraw(False)

        theseKeys = event.getKeys(keyList=['escape', 'q'])
        # check for quit:
        if theseKeys:
            core.quit()

        if listen_keys:
            theseKeys = event.getKeys(keyList=listen_keys)
            if theseKeys:
                set_components_attr(components_list, 'autoDraw', False)
                if countdown_component:
                    countdown_component.setAutoDraw(False)
                win.flip()
                break

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False # will revert to True if at least one component still running
        if components_list:
            for component in components_list:
                if hasattr(component, "status") and component.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
        else:
            if status != FINISHED:
                continueRoutine = True

        # keep mouse in this window
        (x, y) = mouse.getPos()
        if x > 0.5:
            mouse.setPos((0.5, y))
        if x < -0.5:
            mouse.setPos((-0.5, y))

        # refresh the screen
        win.flip()

    return theseKeys, timer.getTime()
