import numpy as np

def save_pa_sim(sim, path):
    sim = (sim * 255).astype(np.uint8)
    np.savez_compressed(path, sim)

def load_pa_sim(path):
    sim = np.load(path)['arr_0'] / 255
    return sim
