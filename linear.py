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
class Linear(AudioModule):
    """Combine multiple inputs linearly into a single output."""

    name = "Linear"
    category = "Channels"
    description = "Linearly combine multiple inputs to a single output."

    def __init__(self,
                 num_inputs: int = 2,
                 **kwargs):
        super().__init__(in_chs=[1]*num_inputs,
                         num_inputs=num_inputs,
                         **kwargs)
        self.amps = np.array(
            [1.0/num_inputs]*self.num_inputs, dtype=self.dtype)

    async def next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        output_chunk = np.zeros(self.chunk_size, dtype=self.dtype)
        for i in range(self.num_inputs):
            output_chunk += self.get_input_chunk(i).buffer[:,0]*self.amps[i]
        self.send_signal(output_chunk.reshape(self.chunk_size,self.out_chs[0]))
        return AM_CONTINUE

    def get_widget_params(self):
        params = {
            'meta_order': ['num_inputs'],
            'num_inputs':{
                'name' : f'Number of inputs',
                'value' : self.num_inputs,
                'type' : 'int',
                'min' : 2,
                'reconfigure' : True
            }
        }
        for i in range(self.num_inputs):
            param = f"amp_{i}"
            params['meta_order'].append(param)
            params[param] = {
                'name': f'Amplitude {i}',
                'value': self.amps[i],
                'type': 'float',
            }
        return super().get_widget_params() | params

    def reconfigure_io(self,num_inputs:int):
        self.configure_io(num_inputs,1,in_chs=[1]*num_inputs,out_chs=[1])
        self.amps = np.array(
            [1.0/num_inputs]*self.num_inputs, dtype=self.dtype)
        #print("reconfigured with inputs",self.num_inputs)

    def set_widget_params(self, params):
        if 'num_inputs' in params:
            if params['num_inputs']['value'] != self.num_inputs:
                self.reconfigure_io(params['num_inputs']['value'])
                self.old_amps = self.amps
                self.amps = np.array([1.0/self.num_inputs]*self.num_inputs, dtype=self.dtype)
                l=min(len(self.amps),len(self.old_amps))
                self.amps[0:l]=self.old_amps[0:l]
        for i in range(self.num_inputs):
            self.amps[i] = params[f'amp_{i}']['value']
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':str(self.amps)
        }
