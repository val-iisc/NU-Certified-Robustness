# evaluate a smoothed classifier on a dataset
import argparse
import os
import numpy as np
#import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify for multiple perturbation bounds')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd_gauss", type=float, help="noise hyperparameter same as train noise_sd_gauss")
parser.add_argument("noise_sd_unif", type=float, help="noise hyperparameter same as train noise_sd_unif")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--start", type=int, default=0, help="starting index of image")
args = parser.parse_args()

if __name__ == "__main__":
    #load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    #create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.noise_sd_gauss, args.noise_sd_unif)

    #prepare output file
    f = open(args.outfile, 'w')
    if (args.noise_sd_gauss == 0 and args.noise_sd_unif != 0): # only Uniform smoothing
        print("idx\tlabel\tfinal_predict\tfinal_l2_radius\tfinal_l1_radius\tfinal_correct\ttime_l1", file = f, flush = True)
    elif (args.noise_sd_gauss != 0 and args.noise_sd_unif == 0): # only Gauss smoothing
        print("idx\tlabel\tfinal_predict\tfinal_l2_radius\tfinal_l1_radius\tfinal_correct\ttime_l2", file = f, flush = True)
    else:
        print("idx\tlabel\tpredict_gauss\tradius_l2_gauss\tradius_l1_gauss\tcorrect_gauss\tpredict_unif\tradius_l1_unif\tradius_l2_unif\tcorrect_unif\tfinal_prediction\tfinal_l2_radius\tfinal_l1_radius\tfinal_correct\time_l2\time_l1", file=f, flush=True)


    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(args.start,len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        # certify the prediction of g around x
        x = x.cuda()

        before_time_l2 = time()
        prediction_l2, radius_l2_gauss = smoothed_classifier.certify_l2(x, args.N0, args.N, args.alpha, args.batch)
        after_time_l2 = time()
        radius_l1_gauss = radius_l2_gauss
        correct_l2 = int(prediction_l2 == label)
        time_elapsed_l2 = str(datetime.timedelta(seconds=(after_time_l2 - before_time_l2)))
        
        before_time_l1 = time()
        prediction_l1, radius_l1_unif = smoothed_classifier.certify_l1(x, args.N0, args.N, args.alpha, args.batch)
        after_time_l1 = time()
        radius_l2_unif = radius_l1_unif/np.sqrt(3072)
        correct_l1 = int(prediction_l1 == label)
        time_elapsed_l1 = str(datetime.timedelta(seconds=(after_time_l1 - before_time_l1)))

        # proposed certification method
        if (prediction_l1 == prediction_l2):
            final_prediction = prediction_l1
        elif (prediction_l1 == -1):
            final_prediction = prediction_l2
        elif (prediction_l2 == -1):
            final_prediction = prediction_l1
        else:
            final_prediction = -1
        
        if (final_prediction == -1):
            final_l2_radius = 0
            final_l1_radius = 0
        else:
            final_l2_radius = np.max(radius_l2_gauss, radius_l2_unif)
            final_l1_radius = np.max(radius_l1_gauss, radius_l1_unif)
        
        correct_final  = int(final_prediction == label)

        
        if (args.noise_sd_gauss == 0 and args.noise_sd_unif != 0):
            print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}".format(
                i, label, prediction_l1, radius_l2_unif, radius_l1_unif, correct_l1,  time_elapsed_l1), file=f, flush=True)
        elif (args.noise_sd_gauss != 0 and args.noise_sd_unif == 0):
            print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}".format(
                i, label, prediction_l2, radius_l2_gauss, radius_l1_gauss, correct_l2,  time_elapsed_l2), file=f, flush=True)
        else:
            print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}".format(
                i, label, prediction_l2, radius_l2_gauss, radius_l1_gauss, correct_l2, prediction_l1, radius_l1_unif, radius_l2_unif, correct_l1, final_prediction, final_l2_radius, final_l1_radius, correct_final,  time_elapsed_l2, time_elapsed_l1), file=f, flush=True)

    f.close()
