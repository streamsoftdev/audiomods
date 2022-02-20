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
class WaveGenerator(AudioModule):
    """Generate a sinusoidal wave."""
    
    name = "Wave Generator"
    category = "Signal source"
    description = "Generate a constant sinusoidal wave = amp * cos(t * 2*pi * freq + phase * pi) + bias"

    def __init__(self,
                 fp=(440.0, 0.0, 0.1, 0.0),
                 shape:str="sinusoid",
                 **kwargs):
        super().__init__(num_inputs=0, **kwargs)
        self.fp = fp  # (freq,phase,amp,bias)
        self.shape=shape
        self.current_idx=0
        self.control = {
            'freq': 'manual',
            'phase': 'manual',
            'amp': 'manual',
            'bias': 'manual'
        }
        self.period_samples=self.sample_rate/self.fp[0]
        self.param_reset()

    async def next_chunk(self):
        if self.num_inputs == 0:
            if self.shape=="sinusoid":
                x = self.fp[2]*np.cos((self.t+self.current_idx)/self.sample_rate
                                    * 2.0*np.pi*self.fp[0]+self.fp[1])+self.fp[3]
            else:
                x = np.cos((self.t+self.current_idx)/self.sample_rate
                                    * 2.0*np.pi*self.fp[0]+self.fp[1])
                x[x<0]=-1
                x[x>=0]=1
                x=x*self.fp[2]+self.fp[3]
            self.current_idx+=self.chunk_size
            self.send_signal(x.reshape(self.chunk_size,1))
            return AM_CONTINUE
        else:
            underrun,cyclic = self.input_underrun()
            if (not self.input_pending()) or underrun:
                if cyclic:
                    return AM_CYCLIC_UNDERRUN
                else:
                    return AM_INPUT_REQUIRED
            input_chunks = []
            for i in range(self.num_channels):
                input_chunks.append(self.get_input_chunk(i).buffer[:,0])
            fpp = [0]*4
            chk_idx = 0
            fp_idx = 0
            for param in ['freq', 'phase', 'amp', 'bias']:
                if self.control[param] == "driven":
                    fpp[fp_idx] = input_chunks[chk_idx]
                    chk_idx += 1
                else:
                    fpp[fp_idx] = self.fp[fp_idx]
                fp_idx += 1

            if self.shape=="sinusoid":
                x = fpp[2]*np.cos((self.t+self.current_idx)/self.sample_rate *
                              2.0*np.pi*fpp[0]+fpp[1])+fpp[3]
            if self.shape=="square":
                x = np.cos((self.t+self.current_idx)/self.sample_rate *
                              2.0*np.pi*fpp[0]+fpp[1])
                x[x<0]=-1
                x[x>=0]=1
                x=x*fpp[2]+fpp[3]

            self.current_time += self.chunk_size
            self.send_signal(x.reshape(self.chunk_size,self.out_chs[0]))
            return AM_CONTINUE

    def param_reset(self):
        self.t = np.arange(self.chunk_size)

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['freq', 'phase', 'amp', 'bias','shape'],
            'freq': {
                'name': 'Frequency (Hz)',
                'value': self.fp[0],
                'type': 'float',
                'min': 0.0,
                'control': self.control['freq']
            },
            'phase': {
                'name': 'Phase (normalized)',
                'value': self.fp[1]/(np.pi),
                'type': 'float',
                'min': -1.0,
                'max': 1.0,
                'control': self.control['phase']
            },
            'amp': {
                'name': 'Amplitude',
                'value': self.fp[2],
                'type': 'float',
                'control': self.control['amp']
            },
            'bias': {
                'name': 'Bias',
                'value': self.fp[3],
                'type': 'float',
                'control': self.control['bias']
            },
            'shape': {
                'name': 'Shape',
                'value': self.shape,
                'type': 'select',
                'options': ['sinusoid','square']
            }
        }

    def reconfigure_io(self):
        driven = 0
        for param in self.control.keys():
            if self.control[param] == "driven":
                driven += 1
        self.configure_io(num_inputs=driven, num_outputs=1,
                          in_chs=[1]*driven, out_chs=[1])
        self.num_channels = driven

    def set_widget_params(self, params: dict):
        if 'freq' in params:
            freq = max([0.0, params['freq']['value']])
        else:
            freq = self.fp[0]
        if 'phase' in params:
            phase = min([1.0, max([-1.0, params['phase']['value']])])*np.pi
        else:
            phase = self.fp[1]
        if 'amp' in params:
            amp = params['amp']['value']
        else:
            amp = self.fp[2]
        if 'bias' in params:
            bias = params['bias']['value']
        else:
            bias = self.fp[3]
        if 'shape' in params:
            shape = params['shape']['value']
        else:
            shape = self.shape
        self.fp = (freq, phase, amp, bias)
        self.shape = shape
        super().set_widget_params(params)
        reconf = False
        for param in ['freq', 'phase', 'amp', 'bias']:
            if param in params and 'control' in params[param]:
                if self.control[param] != params[param]['control']:
                    reconf = True
                self.control[param] = params[param]['control']
        if reconf:
            self.reconfigure_io()

    def reset(self):
        self.current_idx=0

    def close(self):
        super().close()

    def open(self):
        super().open()
        self.reset()

    def get_status(self):
        return {
            'bottom':f'Freq {nice_frequency_str2(self.fp[0])}, Phase {self.fp[1]:.3f}, Amp {self.fp[2]:.3f}, Bias {self.fp[3]:.3f}'
        }
