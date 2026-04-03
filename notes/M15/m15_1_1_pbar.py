# Examples for progress bar and percentage done
# author: sdm

import pyprind as prog                         # packages required
import time

ITERS = 5000000                                # operations to count
SPLITS = 5                                     # split ops into chunks
STEPS = ITERS/SPLITS                           # ops per chunk
PAUSE = 1                                      # seconds to pause between chunks

print('\n')                                    # force blank lines

pbar = prog.ProgBar(ITERS,title='Bar')         # simple progress bar
for index in range(ITERS):                     # track for loop iterations
    pbar.update()

print('\n')                                    # force blank lines

pbar = prog.ProgPercent(ITERS,title='Percent') # simple percentage tracker
for index in range(ITERS):                     # track for loop iterations
    pbar.update()

print('\n')                                    # force blank lines

# use a different character
pbar = prog.ProgBar(ITERS,bar_char='\u2588',title='alt_char')
for index in range(ITERS):                     # track for loop iterations
    pbar.update()

print('\n')                                    # force blank lines

pbar = prog.ProgBar(ITERS,title='Chunks')      # simple progress bar
pbar.update(0,force_flush=True)                # force the 0% display
for index in range(SPLITS):                    # iterate over the chunks
    time.sleep(PAUSE)                          # pause between chunks
    update_str = 'Chunk '+ str(index)          # add a string to the bar
    pbar.update(STEPS,item_id=update_str)      # add the entire chunk at once

print('\n')                                    # force blank lines
