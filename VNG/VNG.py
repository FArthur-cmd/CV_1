import re
from unittest import result
import numpy as np

class VNG:
    '''
    This class is made to make color images from Bayer filter
    which looks like
    R G
    G B
    '''

    # see __calculate_pixel_for
    EPSILON=1e-4

    def __init__(self, image):
        # add padding to make it easier to calculate matrices (so the calculation speed will be higher)
        # 2 symbols from each border will be enough because we need 5x5 windows,
        # where central element will be taken from image itself
        self.image = image
        self.padded_image = np.pad(self.image, 2)

        # allocate separate matrices to simplify readability
        # all indexes are calculated for padded table (2 added symbols to all sides)
        # |_|_|_______|_|_|
        # |_|_|___x___|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up1 = self.padded_image[1:-3, 2:-2]

        # |_|_|_______|_|_|
        # |_|_|_______|x|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up1Right1 = self.padded_image[1:-3, 3:-1]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|x|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Right1 = self.padded_image[2:-2, 3:-1]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|x|_|
        # |_|_|_______|_|_|
        self.Down1Right1 = self.padded_image[3:-1, 3:-1]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|___x___|_|_|
        # |_|_|_______|_|_|
        self.Down1 = self.padded_image[3:-1, 2:-2]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|x|_______|_|_|
        # |_|_|_______|_|_|
        self.Down1Left1 = self.padded_image[3:-1, 1:-3]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|x|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Left1 = self.padded_image[2:-2, 1:-3]

        # |_|_|_______|_|_|
        # |_|x|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up1Left1 = self.padded_image[1:-3, 1:-3]
        
        # second radius
        # |_|_|___x___|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up2 = self.padded_image[:-4, 2:-2]

        # |_|_|_______|x|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up2Right1 = self.padded_image[:-4, 3:-1]

        # |_|_|_______|_|x|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up2Right2 = self.padded_image[:-4, 4:]
        
        # |_|_|_______|_|_|
        # |_|_|_______|_|x|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up1Right2 = self.padded_image[1:-3, 4:]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|x|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Right2 = self.padded_image[2:-2, 4:]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|x|
        # |_|_|_______|_|_|
        self.Down1Right2 = self.padded_image[3:-1, 4:]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|x|
        self.Down2Right2 = self.padded_image[4:, 4:]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|x|_|
        self.Down2Right1 = self.padded_image[4:, 3:-1]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|___x___|_|_|
        self.Down2 = self.padded_image[4:, 2:-2]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|x|_______|_|_|
        self.Down2Left1 = self.padded_image[4:, 1:-3]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|x|_|
        # |_|_|_______|_|_|
        # |x|_|_______|_|_|
        self.Down2Left2 = self.padded_image[4:,:-4]

        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|x|_|
        # |x|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Down1Left2 = self.padded_image[3:-1, :-4]


        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        # |x|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Left2 = self.padded_image[2:-2, :-4]
        
        # |_|_|_______|_|_|
        # |x|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up1Left2 = self.padded_image[1:-3, :-4]

        # |x|_|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|_|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up2Left2 = self.padded_image[:-4,:-4]

        # |_|x|_______|_|_|
        # |_|_|_______|_|_|
        # |_|_|current|x|_|
        # |_|_|_______|_|_|
        # |_|_|_______|_|_|
        self.Up2Left1 = self.padded_image[:-4, 1:-3]

        # Description for gradients
        # Up, Right, Down, Left gradients are calculated in same way for every case (see formulas below)
        self.UpGrad = np.abs(self.Up1 - self.Down1) + \
                      np.abs(self.Up2 - self.image) + \
                      np.abs(self.Up1Left1 - self.Down1Left1)/2 + \
                      np.abs(self.Up1Right1 - self.Down1Right1)/2 + \
                      np.abs(self.Up2Left1 - self.Left1)/2 + \
                      np.abs(self.Up2Right1 - self.Right1)/2

        self.RightGrad = np.abs(self.Right1 - self.Left1) + \
                         np.abs(self.Right2 - self.image) + \
                         np.abs(self.Up1Right1 - self.Up1Left1)/2 + \
                         np.abs(self.Down1Right1 - self.Down1Left1)/2 + \
                         np.abs(self.Up1Right2 - self.Up1)/2 + \
                         np.abs(self.Down1Right2 - self.Down1)/2

        self.DownGrad = np.abs(self.Down1 - self.Up1) + \
                        np.abs(self.Down2 - self.image) + \
                        np.abs(self.Down1Left1 - self.Up1Left1)/2 + \
                        np.abs(self.Down1Right1 - self.Up1Right1)/2 + \
                        np.abs(self.Down2Left1 - self.Left1)/2 + \
                        np.abs(self.Down2Right1 - self.Right2)/2

        self.LeftGrad = np.abs(self.Left1 - self.Right1) + \
                        np.abs(self.Left2 - self.image) + \
                        np.abs(self.Up1Left1 - self.Up1Right1)/2 + \
                        np.abs(self.Down1Left1 - self.Down1Right1)/2 + \
                        np.abs(self.Up1Left2 - self.Up1)/2 + \
                        np.abs(self.Down1Left2 - self.Down1)/2
        
        # other gradients depend on color in center. So lets calculate matricies for 2 cases:
        # color is green
        self.UpRightGreenGrad = np.abs(self.Up1Right1 - self.Down1Left1) + \
                                np.abs(self.Up2Right2 - self.image) + \
                                np.abs(self.Up2Right1 - self.Left1) + \
                                np.abs(self.Up1Right2 - self.Down1)
        self.UpLeftGreenGrad = np.abs(self.Up1Left1 - self.Down1Right1) + \
                               np.abs(self.Up2Left2 - self.image) + \
                               np.abs(self.Up2Left1 - self.Right1) + \
                               np.abs(self.Up1Left2 - self.Down1)
        self.DownRightGreenGrad = np.abs(self.Down1Right1 - self.Up1Left1) + \
                                  np.abs(self.Down2Right2 - self.image) + \
                                  np.abs(self.Down2Right1 - self.Left1) + \
                                  np.abs(self.Down1Right2 - self.Up1)
        self.DownLeftGreenGrad = np.abs(self.Down1Left1 - self.Up1Right1) + \
                                 np.abs(self.Down2Left2 - self.image) + \
                                 np.abs(self.Down2Left1 - self.Right1) + \
                                 np.abs(self.Down1Left2 - self.Up1)
        
        # color is red or blue:
        self.UpRightGrad = np.abs(self.Up1Right1 - self.Down1Left1) + \
                           np.abs(self.Up2Right2 - self.image) + \
                           np.abs(self.Up2Right1 - self.Up1)/2 + \
                           np.abs(self.Up1Right2 - self.Right1)/2 + \
                           np.abs(self.Right1 - self.Down1)/2 + \
                           np.abs(self.Up1 - self.Left1)/2
        self.UpLeftGrad = np.abs(self.Up1Left1 - self.Down1Right1) + \
                          np.abs(self.Up2Left2 - self.image) + \
                          np.abs(self.Up2Left1 - self.Up1)/2 + \
                          np.abs(self.Up1Left2 - self.Left1)/2 + \
                          np.abs(self.Left1 - self.Down1)/2 + \
                          np.abs(self.Up1 - self.Right1)/2
        self.DownRightGrad = np.abs(self.Down1Right1 - self.Up1Left1) + \
                             np.abs(self.Down2Right2 - self.image) + \
                             np.abs(self.Down2Right1 - self.Down1)/2 + \
                             np.abs(self.Down1Right2 - self.Right1)/2 + \
                             np.abs(self.Right1 - self.Up1)/2 + \
                             np.abs(self.Down1 - self.Left1)/2
        self.DownLeftGrad = np.abs(self.Down1Left1 - self.Up1Right1) + \
                            np.abs(self.Down2Left2 - self.image) + \
                            np.abs(self.Down2Left1 - self.Down1)/2 + \
                            np.abs(self.Down1Left2 - self.Left1)/2 + \
                            np.abs(self.Left1 - self.Up1)/2 + \
                            np.abs(self.Down1 - self.Right1)/2


    def __calculate_pixel_for(self, i, j):
        # Let's remember that colours are:
        # R G
        # G B
        current_colour = ''
        
        # see in pattern
        if (i + j) % 2 == 1:
            current_colour = 'G'
        elif i % 2 == 0:
            current_colour = 'R'
        else:
            current_colour = 'B'

        if current_colour == 'G':
            gradients = np.array([self.UpGrad[i, j], self.RightGrad[i, j], self.DownGrad[i, j], self.LeftGrad[i, j],
                                  self.UpRightGreenGrad[i, j], self.UpLeftGreenGrad[i, j], self.DownRightGreenGrad[i, j],
                                  self.DownLeftGreenGrad[i, j]])
        else:
            gradients = np.array([self.UpGrad[i, j], self.RightGrad[i, j], self.DownGrad[i, j], self.LeftGrad[i, j], 
                                  self.UpRightGrad[i, j], self.UpLeftGrad[i, j], self.DownRightGrad[i, j], self.DownLeftGrad[i, j]])
        
        # T = k_1 * Min + k_2 (Max + Min)
        # because k_1 = 1.5, k_2 = 0.5 
        # T = 2 * Min + 0.5 Max
        T = 2 * np.min(gradients) + 0.5 * np.max(gradients)
        
        # because of padding there are some corner cases for gradients
        # if we write many if sections it will slow down work.
        # because of zero values in padding we will recieve min equal to zero
        # if max is not big we will not lose accuracy in next case
        if T < self.EPSILON:
            return [0, 0, 0] 

        choose_gradients = gradients < T
        count = np.sum(choose_gradients)
        current_value = self.image[i, j]
        red_res = 0
        green_res = 0
        blue_res = 0
        
        if current_colour == 'G':
            # green section
            green = np.array([
                    (self.Up2[i, j] + current_value) / 2., 
                    (self.Down2[i, j] + current_value) / 2., 
                    (self.Right2[i, j] + current_value) / 2., 
                    (self.Left2[i, j] + current_value) / 2., 
                    (self.Up2Left2[i, j] + current_value) / 2., 
                    (self.Up2Right2[i, j] + current_value) / 2., 
                    (self.Down2Left2[i, j] + current_value) / 2., 
                    (self.Down2Right2[i, j] + current_value) / 2.
                ])
            
            # there can be two scenarios blue or red. Calculate both than match grads with colours
            # first colour section
            first = np.array([
                self.Up1[i, j], 
                self.Down1[i, j],
                (self.Up1[i, j] + self.Up1Right2[i, j] + self.Down1Right2[i, j] + self.Down1[i, j]) / 4.,
                (self.Up1[i, j] + self.Up1Left2[i, j] + self.Down1Left2[i, j] + self.Down1[i, j]) / 4., 
                (self.Up1[i, j] + self.Up1Right2[i, j]) / 2., 
                (self.Up1[i, j] + self.Up1Left2[i, j]) / 2.,
                (self.Down1[i, j] + self.Down1Right2[i, j]) / 2., 
                (self.Down1[i, j] + self.Down1Left2[i, j]) / 2.
            ])

            # second colour section
            second = np.array([
                self.Right1[i, j],
                self.Left1[i, j],
                (self.Up2Left1[i, j] + self.Up2Right1[i, j] + self.Left1[i, j] + self.Right1[i, j]) / 4.,
                (self.Down2Left1[i, j] + self.Down2Right1[i, j] + self.Left1[i, j] + self.Right1[i, j]) / 4.,
                (self.Right1[i, j] + self.Up2Right1[i, j]) / 2.,
                (self.Right1[i, j] + self.Down2Right1[i, j]) / 2.,
                (self.Left1[i, j] + self.Up2Left1[i, j]) / 2.,
                (self.Left1[i, j] + self.Down2Left1[i, j]) / 2.
            ])
        
            sum_green = np.sum(green * choose_gradients) 
            if (i + 1) % 2 == 0:
                # G R G 
                # B G B
                # G R G
                sum_red = np.sum(first * choose_gradients) 
                sum_blue = np.sum(second * choose_gradients)
            else:
                sum_red = np.sum(second * choose_gradients)
                sum_blue = np.sum(first * choose_gradients)

            red_res = current_value + (sum_red - sum_green) / count
            green_res = current_value
            blue_res = current_value + (sum_blue - sum_green) / count
    
        elif current_colour == 'R':
            # green section
            green = np.array([
                self.Up1[i, j],
                self.Left1[i, j],
                self.Down1[i, j],
                self.Right1[i, j],
                (self.Up1[i, j] + self.Right1[i, j] + self.Up2Right1[i, j] + self.Up1Right2[i, j]) / 4.,
                (self.Down1[i, j] + self.Right1[i, j] + self.Down2Right1[i, j] + self.Down1Right2[i, j]) / 4.,
                (self.Up1[i, j] + self.Left1[i, j] + self.Up2Left1[i, j] + self.Up1Left2[i, j]) / 4.,
                (self.Down1[i, j] + self.Left1[i, j] + self.Down2Left1[i, j] + self.Down1Left2[i, j]) / 4.
            ])
            
            # there can be two scenarios blue or red. Calculate both than match grads with colours
            # first colour section
            first = np.array([
                (self.Up2[i, j] + current_value) / 2.,
                (self.Right2[i, j] + current_value) / 2.,
                (self.Down2[i, j] + current_value) / 2.,
                (self.Left2[i, j] + current_value) / 2.,
                (self.Up2Right2[i, j] + current_value) / 2.,
                (self.Up2Left2[i, j] + current_value) / 2.,
                (self.Down2Right2[i, j] + current_value) / 2.,
                (self.Down2Left2[i, j] + current_value) / 2.
            ])

            # second colour section
            second = np.array([
                (self.Up1Right1[i, j] + self.Up1Left1[i, j]) / 2.,
                (self.Up1Right1[i, j] + self.Down1Right1[i, j]) / 2.,
                (self.Down1Left1[i, j] + self.Down1Right1[i, j]) / 2.,
                (self.Up1Left1[i, j] + self.Down1Left1[i, j]) / 2.,
                self.Up1Right1[i, j],
                self.Down1Right1[i, j],
                self.Down1Left1[i, j],
                self.Up1Left1[i, j]
            ])
            
            sum_green = np.sum(green * choose_gradients) 

            if (i + 1) % 2 == 0:
                # G R G 
                # B G B
                # G R G
                sum_red = np.sum(first * choose_gradients)
                sum_blue = np.sum(second * choose_gradients)

                red_res = current_value
                green_res = current_value + (sum_green - sum_red) / count
                blue_res = current_value + (sum_blue - sum_red) / count
            else:
                sum_red = np.sum(second * choose_gradients)
                sum_blue = np.sum(first * choose_gradients)
                
                red_res = current_value + (sum_red - sum_blue) / count
                green_res = current_value + (sum_green - sum_blue) / count
                blue_res = current_value
        
        return [red_res, green_res, blue_res]

    def process(self):
        ''' Evaluate colour image from filter that was given to class '''
        
        result = []
        # calculate rgb for each pixel
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                result.append(self.__calculate_pixel_for(i, j))

        # return colours
        return 255 - np.array(result).reshape((self.image.shape[0], self.image.shape[1], 3))   