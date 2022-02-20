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
import math

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod


@audiomod
class Stretch(AudioModule):
    """Phase vocoder to stretch signal."""
    
    name = "Stretch"
    category = "Effect"
    description = "Stretch the signal without changing its pitch, using phase vocoding."

    def __init__(self,
                 factor: float = 0.5,
                 window_size: int = 2**13,
                 hop: int = 2**11,
                 use_buffering:bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.window_size = window_size
        self.hop = hop
        self.use_buffering=use_buffering
        self.phase = np.zeros(self.window_size,dtype=self.dtype)
        self.hanning_window = 0.5 * \
            (1.0 - np.cos(2.0*np.pi *
             np.array([x for x in range(self.window_size)],dtype=self.dtype)/self.window_size))
        self.input_idx = 0
        self.output_idx = 0
        self.pre_output_buffer = np.zeros(self.window_size,dtype=self.dtype)
        self.step = int(math.ceil(self.hop*self.factor))
        self.step_correction = self.step - self.hop*self.factor # a small overshoot
        self.step_correction_acc = 0
        self.omega = 2.0*np.pi * \
            np.array([x for x in range(self.window_size)],dtype=self.dtype)/self.window_size
        self.zeros = np.zeros(self.window_size,dtype=self.dtype)
        self.amp = 1.5
        self.epsilon = 1.0
        self.tracking = np.array([False]*self.window_size)
        self.buffering=True

    def get_rate_change(self) -> float:
        return self.factor

    def open(self):
        super().open()
        self.phase = np.zeros(self.window_size)
        self.input_idx = 0
        self.output_idx = 0
        self.pre_output_buffer = np.zeros(self.window_size,dtype=self.dtype)
        self.tracking = np.array([False]*self.window_size)
        self.step_correction_acc = 0
        self.buffering=True

    def princarg(self, phase):
        a = phase/(2.0*np.pi)
        k = np.round(a)
        phase_out = phase-k*2.0*np.pi
        return phase_out

    async def next_chunk(self):
        return self.process_next_chunk()

    def process_next_chunk(self):
        # directly peek into the input buffer
        input_buffer = self.get_in_buf()
        if self.buffering and self.use_buffering:
            self.buffering = input_buffer.size() < self.window_size+10.0*self.step
            return AM_INPUT_REQUIRED
        i2 = self.input_idx*self.hop
        if i2-self.output_idx >= self.chunk_size:
            x = np.split(self.pre_output_buffer, [self.chunk_size])
            output_chunk = x[0]/self.amp
            self.pre_output_buffer = x[1]
            self.send_signal(output_chunk.reshape(self.chunk_size,1))
            self.output_idx += self.chunk_size
            return AM_CONTINUE
        
        while input_buffer.size() >= self.window_size+self.step and input_buffer.size() > self.step:
            self.step_correction_acc+=self.step_correction
            if self.step_correction_acc>=1.0:
                self.step_correction_acc-=1
                step_adjust=1
            else:
                step_adjust=0
            a1 = input_buffer.buffer()[0:self.window_size,0]
            if self.input_idx==0:
                self.pre_output_buffer[:]=a1*self.hanning_window
                self.input_idx+=1
            a2 = input_buffer.buffer()[self.step-step_adjust:self.window_size+self.step-step_adjust,0]
            s1 = np.fft.fft(np.fft.fftshift(self.hanning_window*a1))
            s2 = np.fft.fft(np.fft.fftshift(self.hanning_window*a2))
            
            '''
            Find which frequencies are voiced and which are unvoiced.
            '''
            m1 = np.abs(s2)
            m2 = np.abs(s2)
            voiced1 = m1 >= self.epsilon
            unvoiced1 = ~voiced1
            voiced2 = m2 >= self.epsilon
            unvoiced2 = ~voiced2

            '''
            Calculate the phase.
            '''
            p1 = np.angle(s1)*voiced1
            p2 = np.angle(s2)*voiced2

            '''
            Determine which frequency bins we will start to track and
            which frequency bins we will stop stracking.
            '''
            start_tracking = voiced2 & ~self.tracking
            stop_tracking = ~voiced2 & self.tracking

            # print(np.sum(self.tracking),m2[0],np.max(m2))

            '''
            Compute the phase increment per step for the analysis frames.
            '''
            delta_phi = p2-p1-self.omega*(self.step-step_adjust)
            phase_inc = delta_phi/(self.step-step_adjust) + self.omega

            '''
            Compute the phase of the synthesis frame for those frequencies
            that we were tracking in the last step and didn't stop tracking.
            For frequencies that we started tracking, just copy the phase.
            For frequencies that we stopped tracking, extrapolate the phase
            based on nominal phase change.
            '''
            self.phase = (self.phase+phase_inc*self.hop*(self.tracking & ~stop_tracking) +
                          self.omega*self.hop*stop_tracking) % (2.0*np.pi) + \
                p2*start_tracking

            '''
            Compute the rephased synthesis frame, using only the voiced frequencies
            that we were tracking and didn't stop tracking.
            '''
            a2_rephased = np.fft.ifft(m2*(self.tracking & ~stop_tracking)*np.exp(1.0j*self.phase) +
                                      m2*(start_tracking)*np.exp(1.0j*self.phase)+m2*(stop_tracking)*np.exp(1.0j*self.phase))

            '''
            Remove the phase for frequencies we no longer track and set
            the tracked frequencies to those frequencies that we are now
            tracking.
            '''
            self.phase[unvoiced2] = 0.0
            self.tracking = voiced2

            '''
            Combine the synthesis frame with previous synthesis frames.
            '''
            i2 = self.input_idx*self.hop
            if len(self.pre_output_buffer)+self.output_idx < i2+self.window_size:
                extra = i2+self.window_size - \
                    len(self.pre_output_buffer) - self.output_idx
                self.pre_output_buffer = np.concatenate(
                    [self.pre_output_buffer, np.zeros(extra,dtype=self.dtype)])
            self.pre_output_buffer[i2-self.output_idx:i2-self.output_idx +
                                   self.window_size] += np.fft.fftshift(np.real(a2_rephased))*self.hanning_window
            self.input_idx += 1
            # remove the chunk from the input buffer
            self.get_input_chunk(custom_chunk_size=self.step-step_adjust)
            if i2-self.output_idx >= self.chunk_size:
                x = np.split(self.pre_output_buffer, [self.chunk_size])
                output_chunk = x[0]/self.amp
                self.pre_output_buffer = x[1]
                self.send_signal(output_chunk.reshape(self.chunk_size,self.out_chs[0]))
                self.output_idx += self.chunk_size
                return AM_CONTINUE
        if self.use_buffering:
            self.buffering=True
        x = self.get_in_modules(0)
        if x != None:
            mod,idx = x
            if mod.get_sequence() <= self.sequence:
                return AM_CYCLIC_UNDERRUN
        return AM_INPUT_REQUIRED

    def process_all(self):
        self.buffering=False
        while self.process_next_chunk() == AM_CONTINUE:
            pass

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['factor'],
            'factor': {
                'name': 'Factor',
                'value': self.factor,
                'type': 'float',
                'min': 0.0
            }
        }

    def set_widget_params(self, params):
        if 'factor' in params:
            factor = max([0.0, params['factor']['value']])
        if self.factor != factor:
            self.tracking = np.array([False]*self.window_size)
            self.phase = np.zeros(self.window_size)
        self.factor = factor
        self.step = int(math.ceil(self.hop*self.factor))
        self.step_correction = self.step - self.hop*self.factor # a small overshoot
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':f'Factor {self.factor:.3f}'
        }
