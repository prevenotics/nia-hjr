import argparse


class BaseParams(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParams, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--mode",
            type=str,
            choices=["ViT", "CAF"],
            default="CAF",
            help="network name"
        )
        
        self.add_argument(
            "--dataset",
            type=str,
            default="hjr",
            help="dataset name"
        )

        # self.add_argument(
        #     "--dataset_base_dir",
        #     type=str,
        #     # default=BASE_DIR+"NIA_arg/",
        #     default=DATA_PATH,
        #     help="dataset name"
        # )

        # self.add_argument(
        #     "--image_size",
        #     type=int,
        #     default=512,
        #     help="image size default = 512"
        # )

        self.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Fixed random seed"
        )
        self.add_argument(
            "--local_rank",
            type=int,            
            # required=True,
            help="local rank for DistributedDataParallel"
        )
        self.add_argument(
            "--tensorboard",
            default=True,
            action="store_true"
        )

        self.add_argument(
            "--num_workers",
            type=int,
            default=0,
            # default=0,
            help="num_workers"
        )
        self.add_argument(
            "--patches",
            type=int,
            default=1,
            help="number of patches"
        )
        self.add_argument(
            "--band_patches",
            type=int,
            default=1,
            help="number of related band"
        )


class TrainParser(BaseParams):
    def __init__(self):
        super(TrainParser, self).__init__()
        self.add_argument("--output-dir", default="output", type=str, help="output directory")
        self.add_argument("--img_size", default=512, type=int, help="image_size")
        self.add_argument("--sampling_num", default=64, type=int, help="sampling_point(n x n) for train")
        self.add_argument("--batch_size", type=int, default=32, help="number of batch size")
        self.add_argument("--train_batch_size", default=16, type=int, help="train batch size")
        self.add_argument("--val_batch_size", default=4, type=int, help="validataion batch size")
        self.add_argument("--patch", default=3, type=int)
        self.add_argument("--band_patch", default=3, type=int)
        self.add_argument("--band", default=200, type=int)        
        self.add_argument("--test_freq", type=int, default=5, help="number of evaluation")
        self.add_argument("--epoches", type=int, default=3000, help="epoch number")
        self.add_argument("--save_freq", default=5, type=int, help="save freq")        
        self.add_argument("--eval_freq", default=1, type=int, help="eval freq")                
        self.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate") #0.0005
        self.add_argument("--gamma", type=float, default=0.9, help="gamma")
        self.add_argument("--weight_decay", type=float, default=0, help="weight_decay")
        self.add_argument("--resume", action="store_true", help="resume from checkpoint")
        self.add_argument("--opt", default="adam", type=str, help="nadam, adam")
        self.add_argument("--lrs", default="steplr", type=str, help="cosinealr, steplr")
        # self.add_argument("--enc_dropout", action="store_true", help="dropout for encoder")
        # self.add_argument("--cfg",default="seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_paddle.yaml", help="experiment configure file name",required=True,type=str)
        # self.add_argument("opts",help="Modify config options using the command-line",
        #                 default=None,
        #                 nargs=argparse.REMAINDER)
        
class InferenceParser(BaseParams):
    def __init__(self):
        super(InferenceParser, self).__init__()
        self.add_argument("--output-dir", default="/root/work/src/carbon/outputs_carbon_220822_2_LossWeight_50_005_nan/results_best_checkpoints_loss/", type=str, help="output directory")
        self.add_argument("--val_batch_size", default=1, type=int, help="validataion batch size")
        self.add_argument("--model_path", default="/outputs_carbon_220822_2_LossWeight_50_005_nan/pths/best_checkpoints_loss.pth", type=str, help="pretrained model path")
        # self.add_argument("--image_dir", default="/home/mind2/work/project/dataset/NIA_arg/Validation", type=str, help="image directory")
        self.add_argument("--image_csv", default="/root/work/dataset/test.csv", type=str, help="test image csv")