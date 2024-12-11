

def extract_ood_frame_predictions(
    data_dir: str,
    results_dir: str,
    overwrite: bool,
) -> None:

    pass

    # look for all files that end in _new.csv -> these are OOD labels
    # loop through these
    # for each, load the csv file, and iterate through the rows/index
    #      'labeled-data/<vid_name>/img<#>.png'
    # s = 'labeled-data/vid_name/img0000.png'
    # s2 = '/'.join(s.split('/')[1:])
    # s3 = s2.replace('png', 'mp4')
    # load 51-frame csv file
    # extract center frame
    # put in dataframe
    # save out predictions_<cam_name>.csv
    # compute pixel
