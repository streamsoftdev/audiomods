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

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod


@audiomod
class Downsample(AudioModule):
    name = "Downsample"
    category = "Sampling"
    description = ("Downsample by integral factor M > 1 by keeping "
                   "every M-th sample.")

    def __init__(self,
                 factor: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        # for every sample that we write, we need factor times as many
        self.min_pending = self.chunk_size*self.factor       

    def input_pending(self):
        return self.get_in_buf().size() >= self.min_pending

    def get_rate_change(self) -> float:
        return 1.0/self.factor

    async def next_chunk(self):
        return self.process_next_chunk()

    def process_next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        signal, remaining = self.get_input_chunk(custom_chunk_size=self.min_pending)
        out_chunk = signal[0::self.factor,:]
        self.send_signal(out_chunk)
        return AM_CONTINUE

    def process_all(self):
        #("downsample factor",self.factor)
        while self.process_next_chunk() == AM_CONTINUE:
            pass


    def open(self):
        super().open()

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
        self.factor = int(factor)
        self.min_pending = self.chunk_size*self.factor
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':f'Factor {self.factor:.3f}'
        }