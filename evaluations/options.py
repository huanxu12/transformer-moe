from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)
_data_root = os.path.abspath(os.path.join(file_dir, "..", "data"))


class BotVIOOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="BotVIO options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=_data_root)
        self.parser.add_argument("--eval_data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=_data_root)
        self.parser.add_argument("--pointcloud_path",
                                 type=str,
                                 help="path to the point cloud data",
                                 default=None)
        self.parser.add_argument("--pc_max_points",
                                 type=int,
                                 help="maximum points per point cloud",
                                 default=8192)
        self.parser.add_argument("--pc_min_range",
                                 type=float,
                                 help="minimum radial distance to keep point (meters)",
                                 default=0.5)
        self.parser.add_argument("--pc_max_range",
                                 type=float,
                                 help="maximum radial distance to keep point (meters)",
                                 default=80.0)

        # TRAINING options
        self.parser.add_argument("--train_sequences",
                                 type=str,
                                 help="comma-separated sequence ids for training",
                                 default=None)
        self.parser.add_argument("--finetune_checkpoint",
                                 type=str,
                                 help="path to an existing checkpoint to resume or fine-tune",
                                 default=None)
        self.parser.add_argument("--freeze_visual",
                                 help="freeze visual encoder during training",
                                 action="store_true")
        self.parser.add_argument("--freeze_point",
                                 help="freeze point encoder during training",
                                 action="store_true")
        self.parser.add_argument("--freeze_imu",
                                 help="freeze IMU encoder during training",
                                 action="store_true")
        self.parser.add_argument("--output_checkpoint",
                                 type=str,
                                 help="path to save the trained checkpoint",
                                 default="pretrain_models/multimodal_initial.pth")
        self.parser.add_argument("--resume_optimizer",
                                 help="if set, resume optimizer state from checkpoint",
                                 action="store_true")
        self.parser.add_argument("--log_csv",
                                 type=str,
                                 help="optional CSV file to log epoch metrics",
                                 default=None)
        self.parser.add_argument("--save_every_epoch",
                                 type=int,
                                 help="if >0, save checkpoint every N epochs",
                                 default=0)
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="botvio")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="odom")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay in AdamW",
                                 default=1e-2)
        self.parser.add_argument("--drop_path",
                                 type=float,
                                 help="drop path rate",
                                 default=0.2)
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti_odom",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true",
                                 default='png')
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        self.parser.add_argument("--profile",
                                 type=bool,
                                 help="profile once at the beginning of the training",
                                 default=True)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--train_batch_size",
                                 type=int,
                                 help="batch size for training scripts",
                                 default=2)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of training epochs",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate for optimizers",
                                 default=1e-4)
        self.parser.add_argument("--val_sequences",
                                 type=str,
                                 help="comma-separated sequence ids used for on-the-fly validation during training",
                                 default=None)
        self.parser.add_argument("--val_interval",
                                 type=int,
                                 help="run validation every N epochs (requires --val_sequences)",
                                 default=1)
        self.parser.add_argument("--val_metrics_dir",
                                 type=str,
                                 help="directory to store per-epoch validation metrics JSON",
                                 default=None)
        self.parser.add_argument("--best_checkpoint",
                                 type=str,
                                 help="path to save checkpoint with best validation performance",
                                 default=None)
        self.parser.add_argument("--early_stop_patience",
                                 type=int,
                                 help="epochs without validation improvement before early stopping",
                                 default=0)
        self.parser.add_argument("--early_stop_min_delta",
                                 type=float,
                                 help="minimum improvement in validation metric to reset patience",
                                 default=1e-3)
        self.parser.add_argument("--lr_plateau_factor",
                                 type=float,
                                 help="factor to reduce LR on plateau when validation is enabled",
                                 default=0.3)
        self.parser.add_argument("--lr_plateau_patience",
                                 type=int,
                                 help="patience for LR reduction on plateau (validation only)",
                                 default=2)
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default='../model_weights')
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder"])
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=["euroc", "eigen"],
                                 help="which split to run eval on")

        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--eval_sequences",
                                 type=str,
                                 help="comma-separated sequence ids for evaluation",
                                 default=None)
        self.parser.add_argument("--checkpoint_path",
                                 type=str,
                                 help="path to the model checkpoint for evaluation",
                                 default="pretrain_models/multimodal_initial.pth")
        self.parser.add_argument("--metrics_output",
                                 type=str,
                                 help="optional path to dump evaluation metrics as JSON",
                                 default=None)
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")

        # IMU
        self.parser.add_argument('--imu_dropout',
                                 type=float,
                                 default=0.,
                                 help='dropout for the IMU encoder')
        self.parser.add_argument('--v_f_len',
                                 type=int,
                                 default=512,
                                 help='visual feature length')
        self.parser.add_argument('--i_f_len',
                                 type=int,
                                 default=256,
                                 help='imu feature length')

        self.parser.add_argument('--imu_stats',
                                 type=str,
                                 default=None,
                                 help='optional path to IMU normalization stats (JSON with mean/std)')
        self.parser.add_argument('--imu_gravity_axis',
                                 type=int,
                                 default=None,
                                 help='axis index (0:x, 1:y, 2:z) to subtract gravity from after normalization')
        self.parser.add_argument('--imu_gravity_value',
                                 type=float,
                                 default=9.81,
                                 help='gravity magnitude to subtract when --imu_gravity_axis is set')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options



