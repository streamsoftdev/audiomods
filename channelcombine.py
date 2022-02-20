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
class ChannelCombine(AudioModule):
    """Combine multiple single channels into a multi-channel signal."""
    
    name = "Channel Combine"
    category = "Channels"
    description = ("Combine multiple single-channel inputs into a single multi-channel output. "
                   "For example, to combine a left channel and a right channel into a stero audio signal.")

    def __init__(self,
                 num_inputs: int = 2,
                 **kwargs):
        super().__init__(in_chs=[1]*num_inputs,
                         out_chs=[num_inputs],
                         num_inputs=num_inputs,
                         num_outputs=1,
                         **kwargs)

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
        output = np.column_stack(input_chunks)
        self.send_signal(output)
        #print("channel combine output",len(output))
        return AM_CONTINUE

    