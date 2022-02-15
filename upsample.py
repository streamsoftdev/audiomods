#Copyright 2022 Nathan Harwood
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod


@audiomod
class Upsample(AudioModule):
    name = "Upsample"
    category = "Sampling"
    description = ("Upsample by a integer factor L >= 1 by inserting L-1 zeros between samples,"
        " or using sample and hold if set to true.")

    def __init__(self,
                 factor: int = 2,
                 sample_and_hold:bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.sample_and_hold=sample_and_hold
        self.min_pending=int(self.chunk_size/self.factor)
        self.rem = self.chunk_size % self.factor
        self.inc = (self.factor - self.rem) % self.factor
        self.last_sample=np.array([0]*self.out_chs[0])
        # S00S00S0
        # 0S00S00S
        # 00S00S00
        self.idx=0

    def get_rate_change(self) -> float:
        return self.factor

    def last_step(self):
        if self.idx < self.rem:
            return 1
        else:
            return 0

    def input_pending(self):
        return self.get_in_buf().size() >= self.min_pending + self.last_step()

    async def next_chunk(self):
        return self.process_next_chunk()

    def process_next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        signal,remaining = self.get_input_chunk(custom_chunk_size=self.min_pending+self.last_step())
        out_chunk = self.get_in_buf().get_zero_buffer(self.chunk_size)
        out_chunk[self.idx::self.factor,:] = signal
        if self.sample_and_hold:
            for ch in range(self.out_chs[0]):
                out_chunk[0:self.idx,ch]=self.last_sample[ch]
                for i in range(self.idx,self.chunk_size,self.factor):
                    out_chunk[i:i+self.factor,ch]=out_chunk[i,ch]
        self.idx=(self.idx+self.inc) % self.factor
        self.last_sample=signal[-1,:]
        self.send_signal(out_chunk)
        return AM_CONTINUE

    def process_all(self):
        while self.process_next_chunk() == AM_CONTINUE:
            pass

    def open(self):
        super().open()
        self.idx=0
        self.last_sample=np.array([0]*self.out_chs[0])

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['factor'],
            'factor': {
                'name': 'Factor',
                'value': self.factor,
                'type': 'int',
                'min': 1
            }
        }

    def set_widget_params(self, params):
        if 'factor' in params:
            factor = max([1, params['factor']['value']])
        self.factor = factor
        self.min_pending = int(self.chunk_size/self.factor)
        self.rem = self.chunk_size % self.factor
        self.inc = (self.factor - self.rem) % self.factor
        self.idx = 0
        super().set_widget_params(params)
    
    def get_status(self):
        return {
            'bottom':f'Factor {self.factor:.3f}'
        }