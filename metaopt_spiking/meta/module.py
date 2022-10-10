"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch


class HyperparameterDict(torch.nn.ModuleDict):
    def __init__(self, init_collection):
        """
        Light wrapper to store multiple hyperparameters. It extends a ModuleDict
        and stores the hyperparameters in ParameterLists or ParameterDicts.

        Args:
            init_collection: Dictionary containing dictionaries or lists of tensors
        """
        super().__init__()

        for key, tensor_collection in init_collection.items():
            self.update(key, tensor_collection)

    def update(self, key, tensor_collection):
        """
        Add a new (key, tensor_collection) pair overwriting existing keys with the same key.

        NOTE: We cannot store a single torch.nn.Parameter as it is not a subclass of torch.nn.Module.
              As a workaround just store them in a list with a single entry ¯\_(ツ)_/¯

        Args:
            key: String with the name of the entry
            tensor_collection: Dict or list of tensors to be treated as hyperparameters
        """
        if isinstance(tensor_collection, dict):
            # Store new entry as parameter dictionary
            # NOTE: Need to pass list of tuples to ParameterDict to ensure that the ordering is maintained
            super().update({
                key: torch.nn.ParameterDict([
                    (name, torch.nn.parameter.Parameter(tensor))
                    for name, tensor in tensor_collection.items()
                ])
            })
        elif isinstance(tensor_collection, list):
            # Store new entry as parameter list
            super().update({
                key: torch.nn.ParameterList([
                    torch.nn.parameter.Parameter(tensor)
                    for tensor in tensor_collection
                ])
            })
        else:
            raise ValueError("Type \"{}\" not supported.".format(type(tensor_collection)))
