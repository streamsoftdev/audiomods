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
class Normalize(AudioModule):
    name = "Normalize"
    category = "Filter"
    description = ("Maintain a running maximum of the input signal and produce "
                    "an output signal divided by that maximum.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max = 0.0

    async def next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        signal = self.get_input_chunk().buffer[:,0]
        self.max = max([self.max, np.max(np.abs(signal))])
        if self.max>0.0:
            signal = signal/self.max
        self.send_signal(signal.reshape(self.chunk_size,self.out_chs[0]))
        return AM_CONTINUE

    def open(self):
        super().open()
        self.max = 0.0

    def get_status(self):
        return {
            'bottom':'Peak {self.max:.3f}'
        }
