#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 12:37:24 2019

@author: 2020shatgiskessell
"""
import numpy as np

def is_it_brown (rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]

    # Kind of maximum lightness
    if blue >= 128:
        return False

  # how green or red tinted can it be
    if np.abs(red - green) > 100:
        return False
    if red+green+blue < 30:
        return False
    else:
        return True

is_it_brown ([17,4,4])

