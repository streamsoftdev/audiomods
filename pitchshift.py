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

from audiomodule.audiomodule import AM_CONTINUE, AM_INPUT_REQUIRED, AudioModule, audiomod, Buffer
from audiomods.stretch import Stretch
from audiomods.resample import Resample


@audiomod
class PitchShift(AudioModule):
    """Shift the pitch of the signal without changing its rate."""

    name = "Pitch Shift"
    category = "Musical"
    description = "Shift the pitch of the signal without changing its rate."

    def __init__(self,
                 semitones: float = 1.0,
                 window_size: int = 2**13,
                 stride: int = 2**11,
                 use_buffering:bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.semitones = semitones
        self.window_size = window_size
        self.stride = stride
        self.factor = 2**(1.0*semitones/12.0)
        self.total_consumed=0
        self.total_produced=0
        self.stretchModule = Stretch(
            factor=1.0/self.factor, window_size=self.window_size, hop=self.stride,
            use_buffering=use_buffering)
        self.speedxModule = Resample(1.0/self.factor,polled=True)
        self.stretchModule.connect(self.speedxModule)

    async def next_chunk(self):
        while True:
            if await self.speedxModule.next_chunk() == AM_INPUT_REQUIRED:
                if await self.stretchModule.next_chunk() == AM_INPUT_REQUIRED:
                    if not self.input_pending():
                        return AM_INPUT_REQUIRED
                    else:
                        signal = self.get_in_buf().get_all()
                        #self.total_consumed+=len(signal)
                        self.stretchModule.receive_signal(signal)
                        continue
                else:
                    continue
            else:
                chunk = self.speedxModule.get_out_buf().get_chunk(self.chunk_size).buffer
                self.send_signal(chunk)
                return AM_CONTINUE

    def process_all(self):
        self.stretchModule.receive_signal(self.get_in_buf().get_all())
        self.stretchModule.process_all()
        self.speedxModule.process_all()
        self.send_signal(self.speedxModule.get_out_buf().get_all())

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['semitones'],
            'semitones': {
                'name': 'Semitones',
                'value': self.semitones,
                'type': 'float',
            }
        }

    def set_widget_params(self, params):
        semitones = params['semitones']['value']
        self.semitones = semitones
        self.factor = 2**(1.0*semitones/12.0)
        stretch_params = self.stretchModule.get_widget_params()
        stretch_params['factor']['value'] = 1.0/self.factor
        self.stretchModule.set_widget_params(stretch_params)
        speedx_params = self.speedxModule.get_widget_params()
        speedx_params['factor']['value'] = 1.0/self.factor
        self.speedxModule.set_widget_params(speedx_params)
        super().set_widget_params(params)

    def open(self):
        super().open()
        self.speedxModule.open()
        self.stretchModule.open()

    def close(self):
        super().close()
        self.speedxModule.close()
        self.stretchModule.close()

    def get_status(self):
        return {
            'bottom':f'Semitones {self.semitones:.3f}'
        }
