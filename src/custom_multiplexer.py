# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, Iterator, Optional, Sized, TypeVar, Callable
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data.datapipes.iter.combining import _DemultiplexerIterDataPipe, _ChildDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
from torch.utils.data.datapipes._decorator import functional_datapipe

T_co = TypeVar("T_co", covariant=True)


class SampleMultiplexer(IterDataPipe[T_co]):
    """
    Takes a `Dict` of (IterDataPipe, Weight), and yields items by sampling from these
    DataPipes with respect to their weights. When individual DataPipes are exhausted, continues to sample from
    the remaining DataPipes according to their relative weights.
    If you wish to maintain the same ratio of weights indefinitely, you need to ensure that the
    inputs are never exhausted, by, for instance, applying ``cycle`` to them.

    Sampling is controlled by the provided random ``seed``. If you don't provide it, the sampling
    will not be deterministic.

    Args:
        pipes_to_weights_dict: a `Dict` of IterDataPipes and Weights. The total weight of
            unexhausted DataPipes will be normalized to 1 for the purpose of sampling.
        seed: random seed to initialize the random number generator

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, SampleMultiplexer
        >>> source_dp1 = IterableWrapper([0] * 10)
        >>> source_dp2 = IterableWrapper([1] * 10)
        >>> d = {source_dp1: 99999999, source_dp2: 0.0000001}
        >>> sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d, seed=0)
        >>> list(sample_mul_dp)
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    """

    def __init__(
        self,
        pipes_to_weights_dict: Dict[IterDataPipe[T_co], float],
        batch_size: int,
        seed: Optional[int] = None,
    ):
        if not pipes_to_weights_dict:
            raise ValueError(
                "Empty dictionary passed to SampleMultiplexerDataPipe")
        total_weight: float = 0
        for v in pipes_to_weights_dict.values():
            if v <= 0:
                raise ValueError(
                    f"Expecting a positive and non-zero weight, got {v}")
            total_weight += v

        self.pipes_and_weights = [(k, v / total_weight)
                                  for k, v in pipes_to_weights_dict.items()]
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.batch_size = batch_size
        # self.drop_last = drop_last
        if seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(seed)

    def __iter__(self) -> Iterator[T_co]:
        pipes_and_weights = [(iter(k), v) for k, v in self.pipes_and_weights]
        while len(pipes_and_weights) > 1:
            r = self.random.random()
            s: float = 0
            for it, weight in pipes_and_weights:
                s += weight
                if r < s:
                    try:
                        batch = [next(it) for _ in range(self.batch_size)]
                        flag = True
                        for i in range(1, self.batch_size):
                            if batch[i]["bucket_id"] != batch[i-1]["bucket_id"]:
                                flag = False
                                break
                        if flag:
                            yield batch
                        else:
                            continue
                    except StopIteration:
                        # remove the current stream
                        new_total = 1 - weight
                        assert new_total > 0
                        pipes_and_weights = [(k, v / new_total)
                                             for k, v in pipes_and_weights if k != it]
                    break

        # only one stream left
        while True:
            try:
                batch = [next(pipes_and_weights[0][0])
                         for _ in range(self.batch_size)]
                yield batch
            except StopIteration:
                break

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp, _ in self.pipes_and_weights):
            return sum(len(dp) for dp, _ in self.pipes_and_weights)
        else:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe('mydemux')
class MyDemultiplexerIterDataPipe(IterDataPipe):
    def __new__(cls, datapipe: IterDataPipe, num_instances: int,
                classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool = False, buffer_size: int = 1000):
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instaces` larger than 0, but {num_instances} is found")

        _check_unpickable_fn(classifier_fn)

        # When num_instances == 1, demux can be replaced by filter,
        # but keep it as Demultiplexer for the sake of consistency
        # like throwing Error when classification result is out of o range
        container = _MyDemultiplexerIterDataPipe(
            datapipe, num_instances, classifier_fn, drop_none, buffer_size)
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _MyDemultiplexerIterDataPipe(_DemultiplexerIterDataPipe):
    def _get_max_instance(self):
        max_id = -1
        max_size = -1
        for i in range(self.num_instances):
            if len(self.child_buffers[i]) > max_size:
                max_id = i
                max_size = len(self.child_buffers[i])
        return max_id, max_size

    def _find_next(self, instance_id: int) -> T_co:
        while True:
            if self.main_datapipe_exhausted or self._child_stop[instance_id]:
                raise StopIteration
            if self._datapipe_iterator is None:
                raise ValueError(
                    "_datapipe_iterator has not been set, likely because this private method is called directly "
                    "without invoking get_next_element_by_instance() first.")
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
                StreamWrapper.close_streams(value)
                continue
            if classification is None or classification >= self.num_instances or classification < 0:
                raise ValueError(f"Output of the classification fn should be between 0 and {self.num_instances - 1}. " +
                                 f"{classification} is returned.")
            if classification == instance_id:
                return value
            self.child_buffers[classification].append(value)
            self.current_buffer_usage += 1
            if self.current_buffer_usage/self.num_instances > 50 or self.current_buffer_usage > self.buffer_size*0.75:
                max_id, _ = self._get_max_instance()
                self.current_buffer_usage -= 1
                return self.child_buffers[max_id].popleft()
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(
                    f"DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient.")
