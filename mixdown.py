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
class MixDown(AudioModule):
    """Mix all channels to a single channel."""

    name = "Mix Down"
    category = "Channels"
    description = "Mix all channels to a single channel."

    def __init__(self,num_channels=2,**kwargs):
        super().__init__(in_chs=[num_channels],**kwargs)
        self.weights=[]
        self.control={}
        num_channels = self.get_in_chs(0)
        for i in range(num_channels):
            self.weights.append(1.0/num_channels)
            self.control[f'weight_{i}']='manual'

    async def next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        input_chunks = []
        num_channels = self.get_in_chs(0)
        for i in range(self.num_inputs):
            if i == 0:
                input_chunks.append(self.get_input_chunk(i).buffer)
            else:
                input_chunks.append(self.get_input_chunk(i).buffer[:,0])
        output_chunk = np.zeros(self.chunk_size, dtype=self.dtype)
        drive_bidx=1
        for i in range(num_channels):
            if self.control[f'weight_{i}']=='driven':
                output_chunk += input_chunks[0][:,i]*input_chunks[drive_bidx]
                drive_bidx+=1
            else:
                output_chunk += input_chunks[0][:,i]*self.weights[i]
        self.send_signal(output_chunk.reshape(self.chunk_size,self.out_chs[0]))
        return AM_CONTINUE

    def get_widget_params(self):
        params = {
            'meta_order': []
        }
        for i in range(self.get_in_chs()):
            param = f"weight_{i}"
            params['meta_order'].append(param)
            params[param] = {
                'name': f'Weight {i}',
                'value': self.weights[i],
                'type': 'float',
                'control' : self.control[f'weight_{i}']
            }
        return super().get_widget_params() | params

    def reconfigure_io(self):
        inputs=1
        for param in self.control.keys():
            if self.control[param] == "driven":
                inputs+=1
        self.configure_io(num_inputs=inputs, num_outputs=1,
             out_chs=[1],
             in_chs=[self.get_in_chs()]+[1]*(inputs-1))

    def set_widget_params(self, params):
        reconf=False
        for i in range(self.get_in_chs()):
            param = f"weight_{i}"
            if param in params:
                self.weights[i] = params[param]['value']
                if 'control' in params[param]:
                    if self.control[param] != params[param]['control']:
                        reconf = True
                    self.control[param] = params[param]['control']
        if reconf:
            self.reconfigure_io()
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':str(self.weights)
        }
