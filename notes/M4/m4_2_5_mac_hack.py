# deal with issue of MacOS vs tkinter vs matplotlib

import tkinter 
from tkinter import *
import numpy as np

import matplotlib

# need this BETWEEN import of matplotlib and pyplot!
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

