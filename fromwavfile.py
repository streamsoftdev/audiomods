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
import wave
from pathlib import Path

from audiomodule.audiomodule import AM_INPUT_REQUIRED, AudioModule, sw_dtype, audiomod, nice_channels, nice_frequency_str, hms2_str

import audiomodule.audiomodule as am
import audioengine.audioengine as ae
from audiomods.resample import Resample
from audiomods.iirdesign import IIRDesign


@audiomod
class FromWavFile(AudioModule):
    name = "From WAV File"
    category = "Signal source"
    description = "Read audio from a WAV file."

    def __init__(self, file: str = "",
                 num_channels: int = 2,
                 loop:bool = False,
                 **kwargs):
        super().__init__(num_inputs=0, out_chs=[
            num_channels], **kwargs)
        self.file = file
        self.num_channels = num_channels
        self.loop=loop
        self.ready = True
        self.wf = None
        self.resample_module = None
        self.lpf_module = None
        self.norm = 32768

    def normalize(self,data):
        return data/self.norm

    def decode(self, in_data):
        result = np.frombuffer(in_data, dtype=sw_dtype(self.signal_desc.sampwidth))
        chunk_length = len(result) / self.signal_desc.nchannels
        if chunk_length < self.chunk_size:
            result=np.pad(result,(0,int((self.chunk_size-chunk_length)*self.signal_desc.nchannels)))
            chunk_length = len(result) / self.signal_desc.nchannels
        result = np.reshape(
            result, (int(chunk_length), self.signal_desc.nchannels))
        if self.num_channels > self.signal_desc.nchannels:
            result = np.column_stack([result, self.zeros])
        return self.normalize(result[:, list(range(0,self.num_channels))])

    async def next_chunk(self):
        if not self.wf:
            return am.AM_ERROR
        # continue to read data until a chunk is produced
        # or no more data is available
        while True:
            if self.conversion > 1.0:
                if await self.lpf_module.next_chunk() == AM_INPUT_REQUIRED:
                    if await self.resample_module.next_chunk() == AM_INPUT_REQUIRED:
                        data = self.wf.readframes(self.chunk_size)
                        if data != '':
                            npdata = self.decode(data).astype(self.dtype)
                            self.resample_module.receive_signal(npdata)
                            continue
                        else:
                            return am.AM_COMPLETED
                    else:
                        continue
                else:
                    output_chunk, _ = self.lpf_module.get_out_buf().get_chunk(self.lpf_module.chunk_size)
                    self.send_signal(output_chunk)
                    return am.AM_CONTINUE
            elif self.conversion < 1.0:
                if await self.resample_module.next_chunk() == AM_INPUT_REQUIRED:
                    if await self.lpf_module.next_chunk() == AM_INPUT_REQUIRED:
                        data = self.wf.readframes(self.chunk_size)
                        if data != '':
                            npdata = self.decode(data).astype(self.dtype)
                            self.lpf_module.receive_signal(npdata)
                            continue
                        else:
                            return am.AM_COMPLETED
                    else:
                        continue
                else:
                    output_chunk, _ = self.resample_module.get_out_buf().get_chunk(self.resample_module.chunk_size)
                    self.send_signal(output_chunk)
                    return am.AM_CONTINUE
            else:
                data = self.wf.readframes(self.chunk_size)
                if data != '':
                    npdata = self.decode(data).astype(self.dtype)
                    self.send_signal(npdata)
                    return am.AM_CONTINUE
                else:
                    return am.AM_COMPLETED

    def open(self):
        super().open()
        if self.file != "" and self.file != None:
            try:
                self.wf = wave.open(self.file, 'rb')
                self.signal_desc = self.wf.getparams()
                if self.num_channels > self.signal_desc.nchannels:
                    self.zeros = np.zeros(
                        (self.chunk_size, self.num_channels-self.signal_desc.nchannels), dtype=self.dtype)
                else:
                    self.zeros = None
                self.norm = 2**(self.signal_desc.sampwidth*8-1)
                self.ready = True
                self.conversion = self.sample_rate/self.signal_desc.framerate
               
                if self.conversion > 1.0:  # need to upsample
                    self.resample_module = Resample(self.conversion,
                                                    in_chs=[self.num_channels],
                                                    out_chs=[self.num_channels],
                                                    sample_rate=self.sample_rate)
                    self.lpf_module = IIRDesign(wp=[self.sample_rate/2.0*0.999],
                                                ws=[self.sample_rate/2.0],
                                                in_chs=[self.num_channels],
                                                out_chs=[self.num_channels],
                                                sample_rate=self.sample_rate,
                                                polled=True)
                    self.resample_module.connect(self.lpf_module)
                elif self.conversion < 1.0:  # need to downsample
                    self.resample_module = Resample(factor=self.conversion,
                                                    in_chs=[self.num_channels],
                                                    out_chs=[self.num_channels],
                                                    sample_rate=self.sample_rate,
                                                    polled=True)
                    self.lpf_module = IIRDesign(wp=[self.sample_rate/2.0*0.999],
                                                ws=[self.sample_rate/2.0],
                                                in_chs=[self.num_channels],
                                                out_chs=[self.num_channels],
                                                sample_rate=self.sample_rate,)
                    self.lpf_module.connect(self.resample_module)
            except Exception as e:
                self.observer.put(
                    (ae.MODULE_ERROR, (self.mod_id, f"{self.file} could not be opened. {e}."), None))
                self.ready = False

    def close(self):
        super().close()
        if self.wf:
            self.wf.close()
            self.wf = None

    def reset(self):
        super().reset()
        if self.wf:
            self.wf.rewind()

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['filename'],
            'filename': {
                'name': 'Filename',
                'value': self.file,
                'type': 'read-filename',
                'filetypes': [('WAV files','*.wav *.WAV'),("All files","*.*")]
            }
        }

    def set_widget_params(self, params):
        super().set_widget_params(params)
        if 'filename' in params:
            self.close()
            self.file = params['filename']['value']
            self.open()

    def get_status(self):
        if self.wf:
            ch = nice_channels(self.signal_desc.nchannels)
            status = {
                'topleft':Path(self.file).name,
                'topright':hms2_str(self.signal_desc.nframes/self.signal_desc.framerate),
                'bottom':nice_frequency_str(self.signal_desc.framerate),
                'bottomleft':f"{self.signal_desc.sampwidth*8} bit",
                'bottomright':ch,
            }
        else:
            status = {
                'bottom':"Valid filename required."
            }
        return status