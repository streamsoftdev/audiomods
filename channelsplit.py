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


from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod


@audiomod
class ChannelSplit(AudioModule):
    """Split channels from a multi-channel signal."""

    name = "Channel Split"
    category = "Channels"
    description = "Select channels to output from a multi-channel input."

    def __init__(self,
                 channel_nums: list[int] = [0],
                 input_channels=2,
                 multi_outputs: bool = False,
                 **kwargs):
        if multi_outputs:
            num_outputs = len(channel_nums)
            out_chs = [1]*num_outputs
        else:
            num_outputs = 1
            out_chs = [1]
        super().__init__(in_chs=[input_channels],
                         num_outputs=num_outputs,
                         out_chs=out_chs,
                         **kwargs)
        self.channel_nums = channel_nums
        self.input_channels = input_channels
        self.multi_outputs = multi_outputs

    async def next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        signal = self.get_input_chunk().buffer
        if self.multi_outputs:
            # Send selected channels to individual outputs.
            for i in range(len(self.channel_nums)):
                self.send_signal(signal[:, [self.channel_nums[i]]], i)
        else:
            # Send selected channels to the single output.
            self.send_signal(signal[:, self.channel_nums])
        return AM_CONTINUE

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['ch_nums', 'multi_out'],
            'ch_nums': {
                'name': 'Channel Select',
                'value': self.channel_nums,
                'type': 'intlist',
                'min': 0,
                'max': self.input_channels-1,
                'max_length': max([self.out_chs[0],self.num_outputs]),
                'reconfigure':True
            },
            'multi_out': {
                'name': 'Per-channel output',
                'value': self.multi_outputs,
                'type': 'bool',
                'reconfigure': True
            }
        }

    def reconfigure_io(self):
        if self.multi_outputs:
            num_outputs = len(self.channel_nums)
            out_chs = [1]*num_outputs
        else:
            num_outputs = 1
            out_chs = [len(self.channel_nums)]
        self.configure_io(1, num_outputs, in_chs=[self.input_channels],out_chs=out_chs)

    def set_widget_params(self, params):
        if 'ch_nums' in params:
            self.channel_nums = [int(min([self.input_channels-1, max([0, x])]))
                                 for x in params['ch_nums']['value']]
        if 'multi_out' in params:
            x = self.multi_outputs
            self.multi_outputs = params['multi_out']['value']
            if x != self.multi_outputs:
                self.reconfigure_io()
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':f'Selecting {self.channel_nums}'
        }
