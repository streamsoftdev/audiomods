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
import pyaudio

from audiomodule.audiomodule import AM_CONTINUE, AM_INPUT_REQUIRED, AV_RATE, AudioModule, audiomod, nice_channels, nice_frequency_str


@audiomod
class AudioOut(AudioModule):
    name = "Audio Out"
    category = "Signal out"
    description = "Send the signal to the system's audio out."

    def __init__(self,
                 sample_width: int = 2,
                 channels: int = 2,
                 rate: int = AV_RATE,
                 **kwargs):
        super().__init__(num_outputs=0, in_chs=[channels], **kwargs)
        self.sample_width = sample_width
        self.channels = channels
        self.rate = rate
        self.stream = None
        self.buffer_requirement=self.sample_rate
        self.requires_data = True

    def callback(self, in_data, frame_count, time_info, status):
        signal, bsize = self.get_input_chunk(0, frame_count)
        if len(signal) < frame_count:
            signal = np.concatenate(
                [signal, np.zeros((frame_count-len(signal),self.channels))])
        if bsize >= self.buffer_requirement/2.0:
            self.requires_data = False
        else:
            self.requires_data = True
        return (self.encode(signal), pyaudio.paContinue)

    async def next_chunk(self):
        if self.requires_data:
            return AM_INPUT_REQUIRED
        else:
            return AM_CONTINUE

    def encode(self, signal):
        interleaved = signal.flatten()
        out_data = interleaved.astype(np.float32).tobytes()
        return out_data

    def start(self):
        super().start()
        if self.stream:
            self.stream.start_stream()

    def stop(self):
        super().stop()
        if self.stream:
            self.stream.stop_stream()

    def open(self):
        super().open()
        self.p = pyaudio.PyAudio()
        self.paFormat = pyaudio.paFloat32
        self.stream = self.p.open(
            format=self.paFormat,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback
        )
        self.requires_data = False

    def close(self):
        super().close()
        if self.stream:
            self.stream.close()
            self.p.terminate()

    def get_status(self):
        status = {
            'bottomleft':f"{self.sample_width*8} bit",
            'bottom':nice_frequency_str(self.rate),
            'bottomright':nice_channels(self.channels)
        }
        return status
