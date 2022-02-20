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

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod, nice_frequency_str2


@audiomod
class FrequencyModulation(AudioModule):
    """Frequency modulation."""

    name = "Frequency Modulation"
    category = "Modulation"
    description = "Modulate a carrier signal by an input signal."

    def __init__(self,
                 carrier_freq: float = 440.0,
                 carrier_amp: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.carrier_freq = carrier_freq
        self.carrier_amp = carrier_amp
        self.chunk_time = self.chunk_size/self.sample_rate
        self.total_x = np.zeros(self.chunk_size)
        self.current_chunk:int = 0
        self.control = {
            'carrier_freq': 'manual',
            'carrier_amp': 'manual'
        }
        self.param_reset()

    def open(self):
        super().open()
        self.current_chunk = 0
        self.total_x = np.zeros(self.chunk_size)

    async def next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        input_chunks = []
        for i in range(self.num_inputs):
            input_chunks.append(self.get_input_chunk(i).buffer[:,0])
        fp = [self.carrier_freq, self.carrier_amp]
        fpp = [0]*2
        chk_idx = 1  # chunk 0 is x(t)
        fp_idx = 0
        for param in ['carrier_freq', 'carrier_amp']:
            if self.control[param] == "driven":
                fpp[fp_idx] = input_chunks[chk_idx]
                chk_idx += 1
            else:
                fpp[fp_idx] = fp[fp_idx]
            fp_idx += 1
        self.total_x =  np.modf(np.cumsum(input_chunks[0])+self.total_x[-1])[0]
        
        x = fpp[1]*np.cos((self.t+self.current_chunk*self.chunk_time)*2.0*np.pi *
                          (fpp[0]+input_chunks[0]) + 0.001*2.0*np.pi*self.total_x)
        self.current_chunk += 1
        self.send_signal(x.reshape(self.chunk_size,self.out_chs[0]))
        return AM_CONTINUE

    def param_reset(self):
        self.t = np.array([x/self.sample_rate for x in range(self.chunk_size)])
        self.ct = np.cumsum(self.t)+1.0/self.sample_rate

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['carrier_freq', 'carrier_amp'],
            'carrier_freq': {
                'name': 'Carrier Frequency (Hz)',
                'value': self.carrier_freq,
                'type': 'float',
                'min': 0.0,
                'control': self.control['carrier_freq']
            },
            'carrier_amp': {
                'name': 'Carrier Amplitude',
                'value': self.carrier_amp,
                'type': 'float',
                'control': self.control['carrier_amp']
            }
        }

    def reconfigure_io(self):
        driven = 1
        for param in self.control.keys():
            if self.control[param] == "driven":
                driven += 1
        self.configure_io(num_inputs=driven, num_outputs=1, in_chs=[1]*driven,out_chs=[1])

    def set_widget_params(self, params: dict):
        if 'carrier_freq' in params:
            self.carrier_freq = max([0.0, params['carrier_freq']['value']])
        if 'carrier_amp' in params:
            self.carrier_amp = params['carrier_amp']['value']
        super().set_widget_params(params)
        self.param_reset()
        reconf = False
        for param in ['carrier_freq', 'carrier_amp']:
            if param in params and 'control' in params[param]:
                if self.control[param] != params[param]['control']:
                    reconf = True
                self.control[param] = params[param]['control']
        if reconf:
            self.reconfigure_io()

    def get_status(self):
        return {
            'bottomleft':f'Carrier {nice_frequency_str2(self.carrier_freq)}',
            'bottomright':f'Amp {self.carrier_amp:.3f}'
        }
