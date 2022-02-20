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
import scipy.signal as signal

from audiomodule.audiomodule import AM_CONTINUE, AM_CYCLIC_UNDERRUN, AM_INPUT_REQUIRED, AudioModule, audiomod, MODULE_ERROR

FilterName = {
    "Butterworth": 'butter',
    "Chebyshev I": 'cheby1',
    "Chebyshev II": 'cheby2',
    "Cauer/elliptic": 'ellip',
    "Bessel/Thomson": 'bessel'
}

FilterKey = {
    'butter': "Butterworth",
    'cheby1': "Chebyshev I",
    'cheby2': "Chebyshev II",
    'ellip': "Cauer/elliptic",
    'bessel': "Bessel/Thomson"
}


@audiomod
class IIRDesign(AudioModule):
    """Designed IIR Filter.
    
    Using SciPy's filter design.
    """
    
    name = "IIR Design"
    category = "Filter"
    description = "Infinite Impulse Response filter with design parameters."

    def __init__(self,
                 wp: list[float] = [200],
                 ws: list[float] = [300],
                 gpass: float = 1.0,
                 gstop: float = 40.0,
                 ftype: str = "butter",
                 bias:float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.wp = wp
        self.ws = ws
        self.gpass = gpass
        self.gstop = gstop
        self.ftype = ftype
        self.bias = bias
        if len(wp) == 1:
            self.sos = signal.iirdesign(wp[0], ws[0], gpass, gstop,
                                        ftype=ftype, output="sos", fs=self.sample_rate)
        else:
            self.sos = signal.iirdesign(wp, ws, gpass, gstop,
                                        ftype=ftype, output="sos", fs=self.sample_rate)
        self.zi = []
        for ch in range(self.in_chs[0]):
            self.zi.append(signal.sosfilt_zi(self.sos)*self.bias)

    def process_all(self):
        while self.process_next_chunk() == AM_CONTINUE:
            pass

    async def next_chunk(self):
        return self.process_next_chunk()

    def process_next_chunk(self):
        underrun,cyclic = self.input_underrun()
        if (not self.input_pending()) or underrun:
            if cyclic:
                return AM_CYCLIC_UNDERRUN
            else:
                return AM_INPUT_REQUIRED
        in_chunk = self.get_input_chunk().buffer
        out_chunk = np.zeros(
            (self.chunk_size, self.in_chs[0]), dtype=self.dtype)
        for ch in range(self.in_chs[0]):
            out_chunk[:, ch], self.zi[ch] = signal.sosfilt(
                self.sos, in_chunk[:, ch], zi=self.zi[ch])
        self.send_signal(out_chunk)
        return AM_CONTINUE

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['wp', 'ws', 'gpass', 'gstop', 'ftype'],
            'wp': {
                'name': 'Passband edge frequencies',
                'value': self.wp,
                'type': 'floatlist',
                'min': 0.0
            },
            'ws': {
                'name': 'Stopband edge frequencies',
                'value': self.ws,
                'type': 'floatlist',
                'min': 0.0
            },
            'gpass': {
                'name': 'Passband maximum loss (dB)',
                'value': self.gpass,
                'type': 'float',
                'min': 0.0
            },
            'gstop': {
                'name': 'Stopband minimum attenuation (dB)',
                'value': self.gstop,
                'type': 'float',
                'min': 0.0
            },
            'ftype': {
                'name': 'Filter type',
                'value': FilterKey[self.ftype],
                'type': 'select',
                'options': list(FilterName.keys())
            }
        }

    def set_widget_params(self, params):
        if 'wp' in params:
            wp = params['wp']['value']
        else:
            wp = self.wp
        if 'ws' in params:
            ws = params['ws']['value']
        else:
            ws = self.ws
        if 'gpass' in params:
            gpass = params['gpass']['value']
        else:
            gpass = self.gpass
        if 'gstop' in params:
            gstop = params['gstop']['value']
        else:
            gstop = self.gstop
        if 'ftype' in params:
            ftype = FilterName[params['ftype']['value']]
        else:
            ftype = self.ftype
        try:
            if len(wp) == 1:
                sos = signal.iirdesign(wp[0], ws[0], gpass, gstop,
                                       ftype=ftype, output="sos", fs=self.sample_rate)
            else:
                sos = signal.iirdesign(wp, ws, gpass, gstop,
                                       ftype=ftype, output="sos", fs=self.sample_rate)
            zi = []
            for _ in range(self.in_chs[0]):
                self.zi.append(signal.sosfilt_zi(sos)*self.bias)
        except:
            self.observer.put(
                (MODULE_ERROR, (self.mod_id, "The filter parameters are invalid."), None))
            return
        self.wp = wp
        self.ws = ws
        self.gpass = gpass
        self.gstop = gstop
        self.ftype = ftype
        self.sos = sos
        self.zi = zi
        super().set_widget_params(params)

    def get_status(self):
        return {
            'bottom':FilterKey[self.ftype]
        }
