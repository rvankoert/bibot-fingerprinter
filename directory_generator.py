import cv2
import numpy as np
import keras
import tensorflow as tf
import math
from config import *
import random

from datagenerator import DataGenerator
from export_datagenerator import ExportDataGenerator


class DirectoryGenerator():
    def testGenerator(self,
                      path:str,
                      channels:int,
                      do_binarize_otsu:bool,
                      do_binarize_sauvola:bool,
                      height=51,
                      width=251,
                      sauvola_window=11
                      ):
        from os import walk

        filenames = next(walk(path), (None, None, []))[2]  # [] if no file
        print(filenames)
        partition = {'train': [], 'validation': [], 'test': []}
        testLabels = {}
        counter = 0
        for line in filenames:
            # filename
            fileName = os.path.join(path, line)
            if not os.path.exists(fileName):
                # print(fileName)
                continue
            if line.startswith('.'):
                # print(fileName)
                continue

            counter = counter + 1
            partition['test'].append(fileName)
            testLabels[fileName] = ""
            # if counter == 23:
            #     break
        test_params = {
                      'batch_size': config.BATCH_SIZE,
                      'shuffle': False,
                      'do_binarize_otsu': do_binarize_otsu,
                      'do_binarize_sauvola': do_binarize_sauvola,
                      'height': height,
                      'width': width,
                      'channels': channels,
                      'sauvola_window': sauvola_window
        }
        test_generator = ExportDataGenerator(partition['test'], testLabels, **test_params)

        return test_generator