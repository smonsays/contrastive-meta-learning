"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch


def convert_hdf_to_torch(x_hdf, y_hdf, num_steps, num_units, max_time):
    """
    Convert hdf spiking dataset to torch.Tensor.
    """
    x_torch = []
    time_bins = torch.linspace(0, max_time, num_steps)

    for idx in range(len(x_hdf["units"])):
        times = torch.bucketize(torch.tensor(x_hdf['times'][idx]), time_bins)
        units = torch.tensor(x_hdf['units'][idx].astype(int), dtype=torch.float32)
        values = torch.ones_like(times)
        coo_tensor = torch.sparse_coo_tensor(torch.stack((times, units), dim=0), values, size=[num_steps, num_units], dtype=float)
        x_torch.append(coo_tensor.to_dense())

    x_torch = torch.stack(x_torch, dim=0)
    y_torch = torch.tensor(y_hdf, dtype=int)

    return x_torch, y_torch


def data_from_subset(subset):
    """
    Extract data from torch.utils.data.Subset object.
    """
    return [torch.stack(d) for d in list(zip(*list(subset)))]


def rates_to_latency_code(rates, num_steps=50):
    """
    Convert standardized rates in [0, 1] to a time-to-spike temporal code.
    """
    # NOTE: This function assumes that inputs are standardized to [0,1]
    assert rates.min() >= 0.0 and rates.max() <= 1.0

    # Convert rates to time steps
    spike_steps = torch.floor(rates * (num_steps - 1)).long()

    # Create spikes at specified time steps
    data_size, num_inputs = rates.shape
    spikes = torch.zeros(data_size, num_steps, num_inputs)
    spikes.scatter_(dim=1, index=spike_steps.unsqueeze(1), src=torch.ones_like(spikes))
    # plt.imshow(spikes[0].T, cmap='Greys',  interpolation='nearest')

    return spikes


def rates_to_population_code(rates, num_neurons, num_steps=20, population_var=2e-04):
    """
    Convert standardized rates in [0, 1] to a population spike code using Gaussian tuning curves.
    """
    # NOTE: This function assumes that inputs are standardized to [0,1], i.e.
    #       rates are treated as being normalised by the maximum firing rate
    assert rates.min() >= 0.0 and rates.max() <= 1.0

    # Each neuron has a preferred value (c.f. orientation tuning)
    neuron_values = torch.linspace(0.0, 1.0, num_neurons)

    # Use Gaussian tuning curves to encode firing rates
    neuron_rates = torch.exp(- (neuron_values - rates)**2 / (2 * population_var))

    # Generate spikes from a Bernoulli process given the firing rates
    spikes = torch.bernoulli(neuron_rates.unsqueeze(1).expand(-1, num_steps, -1))
    # plt.matshow(spikes[0])

    return spikes


def split_dataset(dataset, split):
    """
    Split a dataset into two parts (training set and validation set).

    Args:
        dataset: Dataset to be split
        split: Float between 0 and 1 that determines the fraction
            of data to be used for the validiation set
    """
    validation_samples = int(split * len(dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - validation_samples, validation_samples]
    )

    return train_dataset, valid_dataset
