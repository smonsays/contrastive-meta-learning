"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import json
import os
import shutil
import torch
import torchvision


def first_elem(iterator):
    """
    Extract first element of arbitrarily nested iterators.
    """
    try:
        return first_elem(next(iter(iterator)))
    except TypeError:
        return iterator


def is_metric_better(new_value, best_value, max=True):
    """
    Generically compare for best value.
    """
    if best_value is None:
        return True
    elif max:
        return new_value > best_value
    else:
        return new_value < best_value


def module_group_keys(module):
    """
    Determine unique parameter group names of a torch.nn.Module.
    """
    return set(name.split(".")[-1] for name, _ in module.named_parameters())


def ray_to_tensorboard(analysis_df, search_space, path, metric_keys=["test_acc", "train_acc", "valid_acc"]):
    """
    Create tensorboard log of ray results.
    """
    writer = torch.utils.tensorboard.SummaryWriter(path)

    # Only consider runs that haven't been early stopped
    analysis_df = analysis_df[analysis_df['iterations_since_restore'] == analysis_df['iterations_since_restore'].max()]

    # Convert all columns with datatypes tensorboard cannot handle to string
    analysis_df = analysis_df.apply(lambda col: col.apply(lambda x: ','.join(map(str, x)) if not isinstance(x, (int, float, str, bool)) and x is not None else x))

    # Add the runs row by row to tensorboard
    for index, row in analysis_df.iterrows():
        hparam_dict = {
            key.replace("config.", ""): row[key]
            for key in row.keys()
            if "config." in key and key.replace("config.", "") in search_space.keys()
        }
        metric_dict = {key: row[key] for key in metric_keys}
        writer.add_hparams(hparam_dict, metric_dict)

    # Zip the tensorboard logging results and remove the folder to save space
    writer.close()
    zip_and_remove((path))


def save_dict_as_json(config, name, dir):
    """
    Store a dictionary as a json text file.
    """
    with open(os.path.join(dir, name + ".json"), 'w') as file:
        json.dump(config, file, sort_keys=True, indent=4, skipkeys=True,
                  default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


def show_tensor(input):
    """
    Transform tensor into PIL object and show in separate window.
    """
    if input.dim() > 3:
        input = torchvision.utils.make_grid(input, nrow=10)

    image = torchvision.transforms.functional.to_pil_image(input)
    image.show()


def zip_and_remove(path):
    """
    Zip and remove a folder to save disk space.
    """
    shutil.make_archive(path, 'zip', path)
    shutil.rmtree(path)
