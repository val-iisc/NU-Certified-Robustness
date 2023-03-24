import numpy as np
import matplotlib
#import tkinter
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math
import argparse

sns.set()

parser = argparse.ArgumentParser(description='Generating plots')
parser.add_argument('--Gauss_smoothing_file', type = str, help = 'file containing result from Gaussian smoothing')
parser.add_argument('--Unif_smoothing_file', type = str, help = 'file containing result from Uniform smoothing')
parser.add_argument('--Gauss_legend', type = str, help = 'legend for Gaussian smoothing plot')
parser.add_argument('--Unif legend', type = str, help = 'legend for unif smoothing plot')
parser.add_argument('--outfile', type=str, help='path to save the plot')
args = parser.parse_args()

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return ((df['final_correct']>=1) & (df['final_l2_radius'] >= radius)).mean()
        


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=8)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()




if __name__ == "__main__":
    plot_certified_accuracy(
        args.outfile, "put title here", 3.5, [
            Line(ApproximateAccuracy(args.Gauss_smoothing_file), args.Gauss_legend),
            Line(ApproximateAccuracy(args.Unif_smoothing_file), args.Unif_legend) 
        ])