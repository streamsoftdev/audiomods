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

from audiomodule.audiomodule import AM_CONTINUE, AudioModule, audiomod


@audiomod
class WhiteNoise(AudioModule):
    """Generate white noise."""
    
    name = "White Noise"
    category = "Signal source"
    description = "Generate white noise."

    def __init__(self,
                 amp: float = 1.0,
                 **kwargs):
        super().__init__(num_inputs=0, **kwargs)
        self.amp = amp

    async def next_chunk(self):
        x = (np.random.rand(self.chunk_size,self.out_chs[0])-0.5)*2.0*self.amp
        self.send_signal(x)
        return AM_CONTINUE

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['amp'],
            'amp': {
                'name': 'Amplitude',
                'value': self.amp,
                'type': 'float'
            }
        }

    def set_widget_params(self, params):
        if 'amp' in params:
            self.amp = params['amp']['value']
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':f'Amp {self.amp:.3f}'
        }
