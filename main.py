import os

import torch
import train
import mnist
import mnistm

import model
import pseudo_train
import only_sumpooling_train
from utils import get_free_gpu
import argparse

save_name = 'omg'
save_dir = '/mnt/sdd/JINSU/pytorch_DANN/trained_models/'


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DANN")

    parser.add_argument(
        "--gpus",
        dest="gpus",
        help="gpu number",
        default="0",
        type=str
    )

    parser.add_argument(
        "--sum_pooling",
        dest="sum_pooling",
        help="choose height or width or both",
        default='0',
        type=str
    )

    parser.add_argument(
        "--save",
        dest="save",
        help="input model name",
        default="dann",
        type=str
    )

    parser.add_argument(
        "--source",
        dest="source",
        help="choose mnist usps",
        default="mnist",
        type=str
    )

    parser.add_argument(
        "--target",
        dest="target",
        help="choose mnist usps",
        default="mnist",
        type=str
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if args.source == "mnist":
        print("1")
        source_train_loader = mnist.mnist_train_loader

    elif args.source == "usps":
        print("2")
        source_train_loader = usps.usps_train_loader

    if args.target == "mnistm":
        print("3")
        target_train_loader = mnistm.mnistm_train_loader

    elif args.target =="usps":
        print("4")
        target_train_loader = usps.usps_train_loader

    print("Called with args:")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model_dir = save_dir + args.save

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()
        # Discriminator_Conv2d = model.Discriminator_Conv2d().cuda()
        # featuremap = model.ExtractorFeatureMap().cuda()
        sumdiscriminator = model.SumDiscriminator().cuda()

        # train.source_only(encoder, classifier, source_train_loader, target_train_loader)
        # train.dann(featuremap, encoder, classifier, discriminator, source_train_loader, target_train_loader,
        #            args.sum_pooling, sumdiscriminator, model_dir, args.save)

        sum_pooling_mode =0
        if args.sum_pooling == 'height':
            sum_pooling_mode = 1
            print(sum_pooling_mode)
        elif args.sum_pooling == 'width':
            sum_pooling_mode= 2
            print(sum_pooling_mode)
        elif args.sum_pooling == 'both':
            sum_pooling_mode=3
            print(sum_pooling_mode)
        else:
            sum_pooling_mode=0
            print(sum_pooling_mode)

        train.dann(args.source, args.target, encoder, classifier, discriminator, source_train_loader, target_train_loader,
                   sum_pooling_mode, sumdiscriminator, model_dir, args.save)

        # only_sumpooling_train.dann(encoder, classifier, source_train_loader, target_train_loader,
        #            sum_pooling_mode, sumdiscriminator, model_dir, args.save)

        # pseudo_train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, sum_pooling_mode, sumdiscriminator, model_dir, args.save)

    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    main()
