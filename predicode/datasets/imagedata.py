"""Create and handles image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import pandas as pd
import plotnine as gg

def hex_2(integer):
    """Transforms integer between 0 and 255 into two-dimensional hex string.

    Args:
        integer: Integer between 0 and 255.

    Returns:
        Hex string of the integer.
    """
    _hex_2 = hex(integer)[2:]
    if integer < 0 or integer > 16**2-1:
        raise ValueError("Specify integer between 0 and 255.")
    if len(_hex_2) == 1:
        _hex_2 = '0' + str(_hex_2)
    return str(_hex_2)

class ImageData():
    """Image data as a four-dimensional numpy array.

    First dimension contains the different pictures, second and third column
    specify the x and y coordinate entries, respectively, and the fourth column
    contains the R, G, and B channels.

    If you would like to specify a black-white image simply specify all channels
    as the respective intensity.

    TODO(@sflippl) create subclass ImageDataBW.

    Attributes:
        labels: Optional labels for each image, associated with each other by
            row index.
        labeller: A plotnine labeller.
        width: Width of the pictures (in pixels).
        height: Height of the pictures (in pixels).
        n_pictures: Number of pictures.
    """

    @staticmethod
    def _validate_data(data): # Validation should be flexible for inheriting classes pylint:disable=no-self-use
        if data.ndim != 4:
            raise ValueError("Data must be four-dimensional.")
        if data.shape[3] != 3:
            raise ValueError("Fourth dimension must contain precisely an R, G,"
                             " and B channel.")
        return data

    def __init__(self, data, labels=None):
        data = self._validate_data(data)
        self.data = data
        if isinstance(labels, list):
            labels = pd.DataFrame({'label_text': labels})
        self.labels = labels
        if labels is None:
            self.labeller = 'label_value'
        else:
            self.labeller = lambda x: self.labels['label_text'][int(x)]
        self.width = data.shape[1]
        self.height = data.shape[2]
        self.n_pictures = data.shape[0]

    def __str__(self):
        msg = "%d %dx%d-pictures." % (self.n_pictures, self.width, self.height)
        return msg

    def dataframe(self, subset=None, n_random=None):
        """Get data frame of the images.

        This method prepares a dataframe of the image data, which is necessary
        for many subsequent processings steps, in particular regarding
        visualization.

        Args:
            subset: Optional list of picture indices that should be included in
                the dataframe. If specified, n_random will be ignored.
            n_random: Optional number of randomly selected images. If neither
                subset nor n_random are specified, all images will be included.

        Returns:
            A dataframe with the following columns:

            image_id: the index of the pictures according to the original
            object.

            x: the x-coordinate.

            y: the y-coordinate.

            r: the intensity of the red channel (0-255).

            g: the intensity of the green channel (0-255).

            b: the intensity of the blue channel (0-255).

            bw: the intensity of the black-white channel (0-255).
        """
        data = copy.deepcopy(self.data)
        if subset is None:
            if n_random is None:
                subset = range(data.shape[0])
            else:
                subset = np.random.choice(range(data.shape[0]),
                                          size=min(n_random, data.shape[0]),
                                          replace=False)
        data = data[subset, :, :, :]
        image_id = list(
            np.repeat(list(subset), repeats=self.height*self.width)
        )
        flattened_y = list(
            np.repeat(list(range(self.width)), repeats=self.height)
        )*len(subset)
        flattened_x = list(
            range(self.height)
        )*self.width*len(subset)
        dataframe = pd.DataFrame({
            'image_id': np.array(image_id),
            'x': np.array(flattened_x),
            'y': np.array(flattened_y),
            'r': data[:, :, :, 0].flatten(),
            'g': data[:, :, :, 1].flatten(),
            'b': data[:, :, :, 2].flatten(),
            'bw': data.mean(axis=3).flatten()
        })
        return dataframe

    def rgb_dataframe(self, subset=None, n_random=None):
        """Returns dataframe of pictures with RGB string.

        For visualization purposes, it is often useful to have an RGB string in
        the style of 'ff0000' that specifies the color of a particular pixel.
        This function returns the same dataframe as
        :func:`~predicode.Imagedata.dataframe`, but includes the RGB code 'rgb'
        as well as the RGB code for black-white images ('rgb_bw')."""
        dataframe = self.dataframe(subset=subset, n_random=n_random)
        dataframe['rgb'] = np.array([
            '#' +
            hex_2(r) +
            hex_2(g) +
            hex_2(b) for r, g, b in zip(
                dataframe['r'], dataframe['g'], dataframe['b']
            )
        ])
        dataframe['rgb_bw'] = np.array([
            '#' + hex_2(int(bw))*3 for bw in dataframe['bw']
        ])
        return dataframe

    def pictures(self, mode='bw', subset=None, n_random=10):
        """Returns a picture of the selected images.

        Creates either a colored or a black-white picture of the selected
        images.

        Args:
            mode: Should the picture be black-white ('bw') or in color
                ('color')?
            subset: Optional list of picture indices that should be included in
                the dataframe. If specified, n_random will be ignored.
            n_random: Optional number of randomly selected images. If neither
                subset nor n_random are specified, all images will be included.

        Returns:
            A plotnine object including all pictures with their label.

        Raises:
            NotImplementedError: mode must be either 'bw' or 'color'."""
        dataframe = self.rgb_dataframe(subset=subset, n_random=n_random)
        if mode == 'bw':
            fill_key = 'rgb_bw'
        elif mode == 'color':
            fill_key = 'rgb'
        else:
            raise NotImplementedError("Pictures are either in black-white"
                                      "('bw') or in color ('color').")
        picture = (gg.ggplot(dataframe, gg.aes(x='x', y='y', fill=fill_key)) +
                   gg.geom_tile() +
                   gg.theme_void() +
                   gg.theme(legend_position='none') +
                   gg.scale_fill_manual(
                       values={
                           key: key for key in dataframe[fill_key].unique()
                       }
                   ) +
                   gg.facet_wrap('image_id', labeller=self.labeller) +
                   gg.scale_y_reverse() +
                   gg.coord_fixed())
        return picture
