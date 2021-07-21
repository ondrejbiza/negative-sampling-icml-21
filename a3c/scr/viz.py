import argparse
import h5py
import matplotlib.pyplot as plt


def main(args):

    f = h5py.File(args.load_path, "r")

    for ep_idx, ep in f.items():

        print("episode {:s}".format(ep_idx))

        for idx in range(10):

            print("step {:d}".format(idx))
            plt.subplot(1, 2, 1)
            plt.imshow(ep["obs"][idx][3:].transpose((1, 2, 0)))
            plt.subplot(1, 2, 2)
            plt.imshow(ep["next_obs"][idx][3:].transpose((1, 2, 0)))
            plt.pause(0.1)

    f.close()


parser = argparse.ArgumentParser("Visualize the contents of a dataset.")
parser.add_argument("load_path")
main(parser.parse_args())
