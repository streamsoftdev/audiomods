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
from fractions import Fraction
from decimal import Decimal

from audiomodule.audiomodule import AM_CONTINUE, AM_INPUT_REQUIRED, AudioModule, audiomod, Buffer
from mods.upsample import Upsample
from mods.downsample import Downsample


@audiomod
class Resample(AudioModule):
    name = "Resample"
    category = "Sampling"
    description = "Resample by upsampling and downsampling."

    def __init__(self,
                 factor: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.fraction = Fraction(Decimal(str(self.factor))).limit_denominator(100)
        self.upsample_module = Upsample(
            self.fraction.numerator, sample_and_hold=True, in_chs=self.in_chs, out_chs=self.out_chs, 
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size)
        self.downsample_module = Downsample(self.fraction.denominator, in_chs=self.in_chs,
                                            out_chs=self.out_chs, polled=True,
                                            sample_rate=self.sample_rate,
                                            chunk_size=self.chunk_size)
        self.upsample_module.connect(self.downsample_module)
        # a positive correction means we are not consuming samples fast enough
        self.correction = self.chunk_size/self.factor-self.chunk_size/(self.fraction.numerator/self.fraction.denominator)
        self.correction_total = 0
        self.total_consumed=0
        self.total_produced=0

    def get_rate_change(self) -> float:
        return self.factor

    async def next_chunk(self):
        while True:
            if await self.downsample_module.next_chunk() == AM_INPUT_REQUIRED:
                if await self.upsample_module.next_chunk() == AM_INPUT_REQUIRED:
                    if self.input_pending():
                        signal = self.get_in_buf().get_all()
                        
                        #print("got a chunk of size ",len(signal))
                        if self.correction_total >= 1:
                            n=int(self.correction_total)
                            self.correction_total-=n
                            #print(f"{self.correction} dropping {n} samples")
                        else:
                            n=0
                       
                        self.upsample_module.receive_signal(signal[n:,:])
                        continue
                    else:
                        return AM_INPUT_REQUIRED
                else:
                    continue
            else:
                chunk = self.downsample_module.get_out_buf().get_chunk(self.chunk_size).buffer
                self.send_signal(chunk)
               
                self.correction_total += self.correction
                if self.correction_total <= -1:
                    n=-int(self.correction_total)
                    #print(f"{self.correction} injecting {n} samples")
                    self.correction_total+=n
                    signal = self.get_in_buf().get_zero_buffer(n)
                    self.get_in_buf().append(signal)
                   
               
                return AM_CONTINUE

    def process_all(self):
        self.upsample_module.receive_signal(self.get_in_buf().get_all())
        while True:
            if self.upsample_module.process_next_chunk() != AM_CONTINUE:
                break
            if self.downsample_module.process_next_chunk() != AM_CONTINUE:
                continue
        self.downsample_module.process_all()
        self.send_signal(self.downsample_module.get_out_buf().get_all())

    def open(self):
        super().open()
        self.upsample_module.open()
        self.downsample_module.open()
        self.correction_total=0

    def close(self):
        super().close()
        self.upsample_module.close()
        self.downsample_module.close()

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['factor'],
            'factor': {
                'name': 'Factor',
                'value': self.factor,
                'type': 'float',
                'min': 0.0
            }
        }

    def set_widget_params(self, params):
        if 'factor' in params:
            factor = max([0.0, params['factor']['value']])
        self.factor = factor
        self.fraction = Fraction(Decimal(str(self.factor))).limit_denominator(100)
        upsample_params = self.upsample_module.get_widget_params()
        upsample_params['factor']['value']=self.fraction.numerator
        self.upsample_module.set_widget_params(upsample_params)
        downsample_params = self.downsample_module.get_widget_params()
        downsample_params['factor']['value']=self.fraction.denominator
        self.downsample_module.set_widget_params(downsample_params)
        self.correction = self.chunk_size/self.factor-self.chunk_size/(self.fraction.numerator/self.fraction.denominator)
        print(f"correction per chunk {self.correction}")
        #self.correction_total = 0
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':f'Factor {self.factor}'
        }

if __name__ == "__main__":
        resample = Resample(chunk_size=16,polled=True)
        signal = np.array(range(2048))
        resample.receive_signal(signal)
        while resample.next_chunk() != AM_INPUT_REQUIRED:
            print(resample.get_out_buf().get_all().buffer)

