from . import ssl
import numpy as np

def get_indices(selection_method, episode_dict, support_size_max=None,is_inductive=0):
    # random 
    if selection_method == "random":
        ind = np.random.choice(episode_dict["unlabeled"]["samples"].shape[0], 1, replace=False)
    
    # random  imbalanced
    if selection_method == "random_imbalanced":
        ind = np.random.choice(episode_dict["unlabeled"]["samples"].shape[0], 1, replace=False)
    
    # ssl
    if selection_method == "ssl":
        ind = ssl.ssl_get_next_best_indices(episode_dict,is_inductive)

   
    # episode_dict["selected_indices"] = ind
    return ind
