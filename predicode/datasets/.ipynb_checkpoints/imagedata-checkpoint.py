import copy
import numpy as np
import pandas as pd
import plotnine as gg

class ImageData:
    
    def hex_2(integer):
        hex_2 = hex(integer)[2:]
        assert len(hex_2) <= 2
        if len(hex_2) == 1:
            hex_2 = '0' + str(hex_2)
        return str(hex_2)
    
    def __init__(self, data, labels = None):
        """
        Initializes ImageData from a 4-dimensional numpy array (image examples, x axis, y axis, channels).
        A possible second argument ('labels') provides a data frame with labels.
        """
        self.data = data
        self.labels = labels
        if labels is None:
            self.labeller = 'label_value'
        else:
            self.labeller = lambda x: self.labels['label_text'][int(x)]
        self.xdim = data.shape[2]
        self.ydim = data.shape[1]
    
    def dataframe(self, subset = None, n_random = None):
        data = copy.deepcopy(self.data)
        if subset is None:
            if n_random is None:
                subset = range(data.shape[0])
            else:
                subset = np.random.choice(range(data.shape[0]), size = n_random, replace = False)
        data = data[subset,:,:,:]
        image_id = list(np.repeat(list(subset), repeats = self.xdim * self.ydim))
        flattened_y = list(np.repeat(list(range(self.ydim)), repeats = self.xdim))*len(subset)
        flattened_x = list(range(self.xdim))*self.ydim*len(subset)
        n = data.shape[0]
        dataframe = pd.DataFrame({
            'image_id': np.array(image_id),
            'x': np.array(flattened_x),
            'y': np.array(flattened_y),
            'r': data[:,:,:,0].flatten(),
            'g': data[:,:,:,1].flatten(),
            'b': data[:,:,:,2].flatten(),
            'bw': data.mean(axis=3).flatten()
        })
        dataframe['rgb'] = np.array([
            '#' + 
            ImageData.hex_2(r) + 
            ImageData.hex_2(g) + 
            ImageData.hex_2(b) for r, g, b in zip(
                dataframe['r'], dataframe['g'], dataframe['b']
            )
        ])
        dataframe['rgb_bw'] = np.array([
            '#' + ImageData.hex_2(int(bw))*3 for bw in dataframe['bw']
        ])
        return dataframe
    
    def pictures(self, mode = 'bw', subset = None, n_random = 10):
        dataframe = self.dataframe(subset = subset, n_random = n_random)
        if mode == 'bw':
            fill_key = 'rgb_bw'
        elif mode == 'color':
            fill_key = 'rgb'
        else:
            raise NotImplemented("Pictures are either in black-white ('bw') or in color ('color').")
        picture = (gg.ggplot(dataframe, gg.aes(x = 'x', y = 'y', fill = fill_key)) + 
                    gg.geom_tile() + 
                    gg.theme_void() + 
                    gg.theme(legend_position = 'none') + 
                    gg.scale_fill_manual(
                        values = {key: key for key in dataframe[fill_key].unique()}
                    ) + 
                    gg.facet_wrap('image_id', labeller = self.labeller) + 
                    gg.scale_y_reverse() + 
                    gg.coord_fixed())
        return picture