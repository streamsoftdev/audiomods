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

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod, Buffer


@audiomod
class Delay(AudioModule):
    """Introduce a delay to the singal."""

    name = "Delay"
    category = "Effect"
    description = "Introduce a delay in the output signal."

    def __init__(self,
                 delay: float = 1.0,
                 max_delay: float = 10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.max_delay = max_delay
        self.delay_change = delay
        self.delay_buffer = np.zeros(int(self.max_delay*self.sample_rate),dtype=self.dtype)
        self.delay_idx = 0
        self.control = {'delay': "manual"}
        self.delay_manual = np.ones(self.chunk_size,dtype=self.dtype)*self.delay*self.sample_rate

    async def next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        chunk_size = self.chunk_size
        in_chunk = self.get_input_chunk().buffer[:,0]
        if self.control['delay'] == "driven":
            delay_vals = self.get_input_chunk(1).buffer[:,0]
            delay_vals[delay_vals < 0] = 0  # force min
            delay_vals[delay_vals > self.max_delay] = self.max_delay  # for max
            delay_vals *= self.sample_rate
            
        else:
            delay_vals = self.delay_manual
        remaining = len(self.delay_buffer)-self.delay_idx-chunk_size
        if remaining < 0:
            x = np.split(self.delay_buffer, [-remaining])
            self.delay_buffer = np.concatenate([x[1], np.zeros(-remaining)])
            self.delay_idx -= -remaining
        self.delay_buffer[self.delay_idx:self.delay_idx+chunk_size] = in_chunk
        self.delay_idx += chunk_size
        dvals = -delay_vals+self.delay_idx-chunk_size+range(chunk_size)
        output_chunk = self.delay_buffer[dvals.astype(np.int)]
        #print(output_chunk[:])
        self.send_signal(output_chunk.reshape(chunk_size,self.out_chs[0]))
        return AM_CONTINUE

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['delay'],
            'delay': {
                'name': 'Delay (s)',
                'value': self.delay,
                'type': 'float',
                'min': 0.0,
                'max': self.max_delay,
                'control': self.control['delay']
            }
        }

    def reconfigure_io(self):
        if self.control['delay'] == "driven":
            num_inputs = 2
        else:
            num_inputs = 1
        self.configure_io(num_inputs=num_inputs, num_outputs=1, in_chs=[1]*num_inputs,
            out_chs=[1])

    def set_widget_params(self, params):
        if 'delay' in params:
            delay = min([max([0.0, params['delay']['value']]),self.max_delay])
            if 'control' in params['delay']:
                if params['delay']['control'] != self.control['delay']:
                    self.control['delay'] = params['delay']['control']
                    self.reconfigure_io()
        else:
            delay = self.delay
        self.delay = delay
        self.delay_manual = np.ones(self.chunk_size,dtype=self.dtype)*self.delay*self.sample_rate
        super().set_widget_params(params)

    def open(self):
        super().open()
        self.delay_buffer = np.zeros(int(self.max_delay*self.sample_rate),dtype=self.dtype)
        self.delay_idx = 0

    def get_status(self):
        return {
            'bottom':f'Delay {self.delay}'
        }
