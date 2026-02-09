import numpy as np
import tkinter as tk

root = tk.Tk()



root.mainloop()

rng = np.random.default_rng()
print(rng.integers(0, 20, 100))

