from gaze import GazeModel, PolynomialGaze
import os
import cv2
import utils

nb_imgs = 45
imgs = []
dataset = 'pattern1'
print("dataset", dataset)
for i in range(nb_imgs):
    i_path = os.path.join('inputs','images', dataset , f'{i}.jpg' )
    imgs.append(i_path)

pos = utils.load_json(dataset, 'positions')

# print(len(imgs), len(pos))
min_order = 2
max_order = 8
nb_order = (max_order - min_order)
for separator in [20, 25, 30, 35, 40]:
    print("separator", separator)
    model = GazeModel(imgs[:-separator], pos[:-separator])
    pmodel = []
    for o in range(min_order, max_order):
        pmodel.append(PolynomialGaze(imgs[:-separator], pos[:-separator], order=o))

    lr_total = 0
    pl_total = [0] * nb_order

    for i in range(nb_imgs-separator, nb_imgs):
        # print(pos[i])
        lr_result = model.estimate(imgs[i])
        # print("lr_result", lr_result)
        lr_diff = utils.dist_tuple(lr_result, pos[i] )
        # print("lr_diff", lr_diff)
        lr_total += lr_diff
        # print("lr_total", lr_total)

        pl_result = []
        for o in range(nb_order):
            r = pmodel[o].estimate(imgs[i]) 
            pl_result.append(r)

        for o in range(nb_order):
            d = utils.dist_tuple(pl_result[o], pos[i] )
            pl_total[o] += d
            # print(i, o, pl_total[o])


    print("linear regression:", lr_total)
    for o in range(nb_order):
        print("pl order", o+2, pl_total[o])

