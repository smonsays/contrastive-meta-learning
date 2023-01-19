import os
from os import path


from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import ClassSplitter, Categorical, Rotation

from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchmeta.utils.data import BatchMetaDataLoader


def get_subdict(adict, name):
    if adict is None:
        return adict
    tmp = {k[len(name) + 1:]:adict[k] for k in adict if name in k}
    return tmp

def save_performance_summary(args):

    # save some stuff in a pickle for later 
    train_epochs = args.epochs

    tp = dict()
    
    tp["best_acc_epoch"] = args.best_acc_epoch
    tp["best_acc"] = args.best_acc
    tp["end_acc"] = args.end_acc 
    tp["finished"] = 1

    # Note, the keywords of this dictionary are defined by the array:
    #   hpsearch._SUMMARY_KEYWORDS

    with open(os.path.join(args.out_dir, hpsearch._SUMMARY_FILENAME), 'w') as f:

        for kw in hpsearch._SUMMARY_KEYWORDS:
            if kw == 'num_train_epochs':
                f.write('%s %d\n' % ('num_train_epochs', train_epochs))
                continue
            else:
                try:
                    f.write('%s %f\n' % (kw, tp[kw]))
                except:
                    f.write('%s %s\n' % (kw, tp[kw]))

class Flip(object):
    def __call__(self,img):
        return 1 - img

def load_data(args):
    meta_dataloader={}
    #-------------Load data----------------------
    if args.dataset=="MiniImagenet":
        dataset_transform = ClassSplitter(shuffle=True,
                                    num_train_per_class=args.num_shots_train,
                                    num_test_per_class=args.num_shots_test)

        transform = Compose([Resize(84), ToTensor(),
                             Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        meta_train_dataset = MiniImagenet("data_i",
                                    transform=transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)
                                          
        meta_val_dataset = MiniImagenet("data_i",
                                    transform=transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_val=True,
                                    dataset_transform=dataset_transform,
                                    download=True)

        meta_test_dataset = MiniImagenet("data_i",
                                    transform=transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_test=True,
                                    dataset_transform=dataset_transform,
                                    download=True)


        meta_dataloader["train"] = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["val"]=BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["test"]=BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        feature_size=5*5*args.hidden_size
        input_channels=3

    if args.dataset=="Omniglot":
        dataset_transform = ClassSplitter(shuffle=True,
                                    num_train_per_class=args.num_shots_train,
                                    num_test_per_class=args.num_shots_test)

        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot("data_o",
                                transform=transform,
                                target_transform=Categorical(args.num_ways),
                                num_classes_per_task=args.num_ways,
                                meta_train=True,
                                use_vinyals_split=False,
                                class_augmentations=[Rotation([90, 180, 270])],
                                dataset_transform=dataset_transform,
                                download=True)
                                    
        meta_test_dataset = Omniglot("data_o",
                                transform=transform,
                                target_transform=Categorical(args.num_ways),
                                num_classes_per_task=args.num_ways,
                                meta_test=True,
                                use_vinyals_split=False,
                                dataset_transform=dataset_transform)

        meta_dataloader["train"] = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["test"]=BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
        feature_size=args.hidden_size
        input_channels=1

    return meta_dataloader, feature_size, input_channels
