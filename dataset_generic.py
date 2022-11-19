import cv2

from config import *

from datagenerator import DataGenerator


class DatasetGeneric():
    def generator(self,
                  imgSize,
                  file,
                  prefix="./",
                  batch_size=1,
                  channels=3,
                  do_binarize_otsu=False,
                  do_binarize_sauvola=False,
                  augment=False,
                  sauvola_window=11,
                  sauvola_k=0.2,
                  minimum_width=0,
                  random_crop=False,
                  cache_path=None,
                  center_zero=False,
                  use_existing_lmdb=None
    ):
        partition = {'train': []}
        trainLabels = {}
        f = open(file)
        counter_small = 0
        counter = 0
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split('\t')
            assert len(lineSplit) >= 2

            # filename
            fileName = os.path.join(prefix,  lineSplit[0])
            if not os.path.exists(fileName):
                # print(fileName)
                continue
            if minimum_width > 0:
                img = cv2.imread(fileName)
                tmp_height, tmp_width, tmp_channels = img.shape
                if tmp_height < 20 or tmp_width < minimum_width or tmp_width / tmp_height < 4:
                    # print("too small: "+fileName)
                #     os.remove(fileName)
                    counter_small = counter_small + 1
                    continue
            label = lineSplit[1]

            counter = counter + 1
            # if (counter > 100):
            #     break
            # # print (label)
            # put sample into list
            partition['train'].append(fileName)
            trainLabels[fileName] = label

        print('skipped ' +str(counter_small) +' too small files')
        print('found ' +str(counter) +' usable files')
        trainParams = {'dim': (imgSize[0], imgSize[1]),
                       'batch_size': batch_size,
                       'shuffle': True,
                       'channels': channels,
                       'do_binarize_otsu': do_binarize_otsu,
                       'do_binarize_sauvola': do_binarize_sauvola,
                       'augment': augment,
                       'sauvola_window': sauvola_window,
                       'sauvola_k': sauvola_k,
                       'random_crop': random_crop,
                       'cache_path': cache_path,
                       'center_zero': center_zero,
                       'use_existing_lmdb': use_existing_lmdb
                       }
        generator = DataGenerator(imgSize, partition['train'], trainLabels, **trainParams)

        return generator