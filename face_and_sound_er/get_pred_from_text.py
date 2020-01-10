import numpy as np
import argparse


def main():
    movies_names = ['avengers', 'erin', 'inception', 'mi_ii', 'wc']
    th = args.threshold
    labels_map = {0: "anger", 1: "anticipation", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise", 7: "trust"}

    for movie in movies_names:
        prob_file = 'movie_subtitles_prediction/' + movie + '_results_prob.npy'
        prob = np.load(prob_file)
        pred = np.argmax(prob, axis=1)
        strongness = np.max(prob, axis=1)
        # pred_file = 'movie_subtitles_prediction/' + movie + '_prediction.npy'
        # strongness_file = 'movie_subtitles_prediction/' + movie + '_strongness.npy'
        # np.save(pred_file, pred)
        # np.save(strongness_file, strongness)
        th_gt_index = np.argwhere(strongness > th)
        th_gt_index = th_gt_index.squeeze()
        th_gt_prediction = pred[th_gt_index]
        labels, counts_elements = np.unique(th_gt_prediction, return_counts=True)
        print(movie)
        print("total strong labels with threshold", th, " : ", th_gt_prediction.shape[0])
        print("total elements", pred.shape[0])
        for i in range(0,len(labels)):
            print(labels[i], labels_map[labels[i]], counts_elements[i])
        print("=======================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    main()
