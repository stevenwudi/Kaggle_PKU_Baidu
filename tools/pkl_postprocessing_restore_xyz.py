"""
YYJ postprocessing--> using multiprocessing
"""
import mmcv

from mmdet.datasets.kaggle_pku_utils import filter_output
from mmdet.utils.plot_mesh_postprocessing import Plot_Mesh_Postprocessing


def filter_output_pool(t):
    return filter_output(*t)


def main():

    pkl_file = '/data/Kaggle/wudi_data/test_Jan29-00-02_epoch_261.pkl'
    outputs = mmcv.load(pkl_file)
    plot_mesh = Plot_Mesh_Postprocessing(car_model_json_dir='/data/Kaggle/pku-autonomous-driving',
                                         test_image_folder='/data/Kaggle/pku-autonomous-driving/test_images')
    outputs_refined = plot_mesh.visualise_pred_postprocessing_multiprocessing(outputs)
    mmcv.dump(outputs_refined, pkl_file[:-4]+'_refined.pkl')

    print("Finished")


if __name__ == '__main__':
    main()
