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

from pathlib import Path
import struct

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_ERROR, AM_INPUT_REQUIRED, MODULE_ERROR, \
    AudioModule, sw_dtype, audiomod, nice_channels, nice_frequency_str, hms2_str


@audiomod
class ToWavFile(AudioModule):
    name = "To WAV File"
    category = "Signal out"
    description = "Write the signal to a WAV file. Supports incremental writing to the WAV file."

    def __init__(self, file: str = "",
                 num_channels: int = 2,
                 sample_width: int = 2,
                 **kwargs):
        super().__init__(num_outputs=0, in_chs=[num_channels], **kwargs)
        self.file = file
        self.num_channels = num_channels
        self.ready = True
        self.wf = None
        self.sample_width=sample_width
        self.bytes_written=0
        self.frames_written=0
        self.requires_data = False
        self.set_write_dtype()

    def set_write_dtype(self):
        if self.sample_width == 1:
            self.write_dtype = np.int8
            self.out_max = (2**7)-1
        elif self.sample_width == 2:
            self.write_dtype = np.int16
            self.out_max = (2**15)-1
        elif self.sample_width == 4:
            self.out_max = 1.0
            self.write_dtype = np.float32

    
    async def next_chunk(self):
        return await self.process_next_chunk()

    def process_next_chunk(self):
        if not self.wf:
            return AM_ERROR
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:   
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        chunk = self.get_input_chunk().buffer
        self.write_chunk(chunk)
        return AM_CONTINUE

    def write_chunk(self,chunk):
        chunk *= self.out_max
        interleaved = chunk.flatten()
        out_data = interleaved.astype(self.write_dtype).tobytes()
        self.frames_written+=len(chunk)
        self.bytes_written+=len(chunk)*self.sample_width*self.num_channels
        self.wf.write(out_data)
        self.update_wav_header()

    def update_wav_header(self):
        current_pos = self.wf.tell()
        self.wf.seek(0)
        WAVE_FORMAT_PCM = 0x0001
        bytes_to_add = b'RIFF'
        
        _datalength = self.frames_written * self.num_channels * self.sample_width

        bytes_to_add += struct.pack('<L4s4sLHHLLHH4s',
            36 + _datalength, b'WAVE', b'fmt ', 16,
            WAVE_FORMAT_PCM, self.num_channels, int(self.sample_rate),
            self.num_channels * int(self.sample_rate) * self.sample_width,
            self.num_channels * self.sample_width,
            self.sample_width * 8, b'data')

        bytes_to_add += struct.pack('<L', _datalength)

        self.wf.write(bytes_to_add)
        self.wf.seek(current_pos)

    def open(self):
        super().open()
        if self.file != "" and self.file != None:
            try:
                self.wf = open(self.file, 'wb')
                self.update_wav_header()
                self.ready = True
            except Exception as e:
                self.observer.put(
                    (MODULE_ERROR, (self.mod_id, f"{self.file} could not be written to. {e}."), None))
                self.ready = False

    def close(self):
        if self.get_in_buf(0).size()>0 and self.wf:
            chunk = self.get_in_buf(0).get_all()
            self.write_chunk(chunk)
        super().close()
        if self.wf:
            self.wf.close()
            self.wf = None

    def reset(self):
        super().reset()
        if self.wf:
            self.wf.seek(0)
            self.bytes_written=0
            self.frames_written=0
            self.update_wav_header()

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['filename'],
            'filename': {
                'name': 'Filename',
                'value': self.file,
                'type': 'write-filename',
                'filetypes': [('WAV files','*.wav *.WAV'),("All files","*.*")],
                'defaultextension': '.wav'
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
            ch = nice_channels(self.num_channels)
            status = {
                'status':'ready',
                'topleft':Path(self.file).name,
                'topright':hms2_str(self.frames_written/self.sample_rate),
                'bottom':nice_frequency_str(self.sample_rate),
                'bottomleft':f"{self.sample_width*8} bit",
                'bottomright':ch,
            }
        else:
            status = {
                'status':'error',
                'bottom':"Valid filename required."
            }
        return status