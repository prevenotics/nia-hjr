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

        self.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Fixed random seed"
        )

        self.add_argument(
            "--tensorboard",
            default=True,
            action="store_true"
        )

        self.add_argument(
            "--num_workers",
            type=int,
            default=8,
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
        self.add_argument("--output-dir", default="output_231031", type=str, help="output directory")
        self.add_argument("--img_size", default=512, type=int, help="image_size")
        self.add_argument("--sampling_num", default=64, type=int, help="sampling_point(n x n) for train")
        self.add_argument("--batch_size", type=int, default=16, help="number of batch size")
        self.add_argument("--train_batch_size", default=16, type=int, help="train batch size")
        self.add_argument("--val_batch_size", default=16, type=int, help="validataion batch size")
        self.add_argument("--patch", default=3, type=int)
        self.add_argument("--band_patch", default=3, type=int)
        self.add_argument("--band", default=200, type=int)        
        self.add_argument("--test_freq", type=int, default=5, help="number of evaluation")
        self.add_argument("--epoches", type=int, default=3000, help="epoch number")
        self.add_argument("--save_freq", default=5, type=int, help="save freq")        
        self.add_argument("--eval_freq", default=1, type=int, help="eval freq")                
        self.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
        self.add_argument("--gamma", type=float, default=0.9, help="gamma")
        self.add_argument("--weight_decay", type=float, default=0, help="weight_decay")
        self.add_argument("--resume", action="store_true", help="resume from checkpoint")
        self.add_argument("--opt", default="adam", type=str, help="nadam, adam")
        self.add_argument("--lrs", default="steplr", type=str, help="cosinealr, steplr")
        
        
class InferenceParser(BaseParams):
    def __init__(self):
        super(InferenceParser, self).__init__()
        self.add_argument("--output-dir", default="/root/work/src/carbon/outputs/", type=str, help="output directory")
        self.add_argument("--val_batch_size", default=1, type=int, help="validataion batch size")
        self.add_argument("--model_path", default="/outputs/pths/best_checkpoints_loss.pth", type=str, help="pretrained model path")
        self.add_argument("--image_csv", default="/root/work/dataset/test.csv", type=str, help="test image csv")