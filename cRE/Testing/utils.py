

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import matplotlib.patches as mpatches

def make_file_adj_graph(edge_mat, graph_title, path_out):
  print("Plot section...")
  values_mat = np.unique(edge_mat.ravel())
  color_dict = {-1: 'lightblue',
              0: 'darkslateblue',
              1: 'yellow'}
  colors=[color_dict[val] for val in list(values_mat)]
  im = plt.imshow(edge_mat,interpolation="none",cmap=ListedColormap(colors))
  plt.title(graph_title)
  colors = [ im.cmap(im.norm(value)) for value in values_mat]
  patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values_mat[i]) ) for i in range(len(values_mat)) ]
  plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
  plt.show()
  # plt.savefig(path