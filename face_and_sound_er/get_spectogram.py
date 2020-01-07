from spectrogram_voxceleb import *
import cv2


def get_wave_file_path_for_video(video_path):
    return video_path.split(".")[0] + ".wav"


def get_window_of_spectrum(video_path, video_point, window_length):
    if video_point > window_length:
        raise("not right video point")
    wav_file_path = get_wave_file_path_for_video(video_path)
    spec_numpy = get_spectrum(wav_file_path)
    video_length = spec_numpy.shape[0]
    if window_length > video_length:
        raise ("frame greater than video length")
    half_window = int((window_length)/2)
    starting_point = video_point - half_window
    ending_point = video_point + half_window
    if starting_point < 0:
        return spec_numpy[0:window_length, :]
    if ending_point >= video_length:
        return spec_numpy[-window_length:, :]
    return spec_numpy[starting_point:ending_point,:]


if __name__ == '__main__':

    wav_file_path = '/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_4.wav'
    spec_img = get_spectrum(wav_file_path)
    spec_img = spec_img[-10:-1,:]
    print(spec_img.shape)
    cv2.imshow('Color image', spec_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #graph_spectrogram(wav_file_path, spec_img)