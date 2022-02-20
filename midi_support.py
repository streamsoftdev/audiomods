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
import os
import subprocess
import wave
import json
import time

from audiomodule.audiomodule import AM_CONTINUE, AV_RATE, sw_dtype
from mods.resample import Resample
from mods.iirdesign import IIRDesign
from mods.towavfile import ToWavFile
from mods.pitchshift import PitchShift

FUNDAMENTAL_NOTE_NAMES = ["A","As","B","C","Cs","D","Ds","E","F","Fs","G","Gs"]

FUNDAMENTAL_NOTE_NAMES_ALT = ["A","Bb","B","C","Db","D","Eb","E","F","Gb","G","Ab"]

NOTE_NUMS = {}
NOTE_NAMES = {}

NOTE_NUMS_ALT = {}
NOTE_NAMES_ALT = {}

name_idx = 3
octave = -1
for i in range(128):
    NOTE_NUMS[f"{FUNDAMENTAL_NOTE_NAMES[name_idx]}{octave}"] = i
    NOTE_NAMES[i]=f"{FUNDAMENTAL_NOTE_NAMES[name_idx]}{octave}"
    name_idx=(name_idx+1) % 12
    if name_idx==3:
        octave+=1
        
name_idx = 3
octave = -1
for i in range(128):
    NOTE_NUMS_ALT[f"{FUNDAMENTAL_NOTE_NAMES_ALT[name_idx]}{octave}"] = i
    NOTE_NAMES_ALT[i]=f"{FUNDAMENTAL_NOTE_NAMES_ALT[name_idx]}{octave}"
    name_idx=(name_idx+1) % 12
    if name_idx==3:
        octave+=1

# Velocity nominal values from:
# http://www.music-software-development.com/midi-tutorial.html
        
VELOCITY = {
    8:'pianissississimo',
    20:'pianississimo',
    31:'pianissimo',
    42:'piano',
    53:'mezzo-piano',
    64:'mezzo-forte',
    80:'forte',
    96:'fortissimo',
    112:'fortississimo',
    127:'fortissississimo'
}

VELOCITY_SHORT = {
    'pppp':8,
    'ppp':20,
    'pp':31,
    'p':42,
    'mp':53,
    'mf':64,
    'f':80,
    'ff':96,
    'fff':112,
    'ffff':127
}

VELOCITY_CONV = {
    'pianissississimo':'pppp',
    'pianississimo':'ppp',
    'pianissimo':'pp',
    'piano':'p',
    'mezzo-piano':'mp',
    'mezzo-forte':'mf',
    'forte':'f',
    'fortissimo':'ff',
    'fortississimo':'fff',
    'fortissississimo':'ffff'
}

NEAREST_VELOCITY={
            'pianissississimo':['pianississimo','pianissimo','piano','mezzo-piano','mezzo-forte','forte','fortissimo','fortississimo','fortissississimo'],
            'pianississimo':['pianissississimo','pianissimo','piano','mezzo-piano','mezzo-forte','forte','fortissimo','fortississimo','fortissississimo'],
            'pianissimo':['pianississimo','piano','pianissississimo','mezzo-piano','mezzo-forte','forte','fortissimo','fortississimo','fortissississimo'],
            'piano':['pianissimo','mezzo-piano','pianississimo','mezzo-forte','pianissississimo','forte','fortissimo','fortississimo','fortissississimo'],
            'mezzo-piano':['piano','mezzo-forte','pianissimo','forte','pianississimo','fortissimo','pianissississimo','fortississimo','fortissississimo'],
            'mezzo-forte':['forte','mezzo-piano','fortissimo','piano','fortississimo','pianissimo','fortissississimo','pianississimo','pianissississimo'],
            'forte':["fortissimo","mezzo-forte","fortississimo","mezzo-piano","fortissississimo","piano","pianissimo","pianississimo","pianissississimo"],
            'fortissimo':['fortississimo','forte','fortissississimo',"mezzo-forte","mezzo-piano","piano","pianissimo","pianississimo","pianissississimo"],
            'fortississimo':['fortissississimo','fortissimo','forte',"mezzo-forte","mezzo-piano","piano","pianissimo","pianississimo","pianissississimo"],
            'fortissississimo':['fortississimo','fortissimo','forte',"mezzo-forte","mezzo-piano","piano","pianissimo","pianississimo","pianissississimo"]
        }

NEAREST_VELOCITY_SHORT = {
            'pppp':['ppp','pp','p','mp','mf','f','ff','fff','ffff'],
            'ppp':['pppp','pp','p','mp','mf','f','ff','fff','ffff'],
            'pp':['ppp','p','pppp','mp','mf','f','ff','fff','ffff'],
            'p':['pp','mp','ppp','mf','pppp','f','ff','fff','ffff'],
            'mp':['p','mf','pp','f','ppp','ff','pppp','fff','ffff'],
            'mf':['f','mp','ff','p','fff','pp','ffff','ppp','pppp'],
            'f':["ff","mf","fff","mp","ffff","p","pp","ppp","pppp"],
            'ff':['fff','f','ffff',"mf","mp","p","pp","ppp","pppp"],
            'fff':['ffff','ff','f',"mf","mp","p","pp","ppp","pppp"],
            'ffff':['fff','ff','f',"mf","mp","p","pp","ppp","pppp"]
}

VELOCITY_VALS = list(VELOCITY.keys())
VELOCITY_VALS.sort()
VELOCITY_NAMES = []
for i in range(128):
    nearest_velocity=VELOCITY_VALS[np.abs(np.array(VELOCITY_VALS,dtype=np.int64)-i).argmin()]
    VELOCITY_NAMES.append(VELOCITY_CONV[VELOCITY[nearest_velocity]])

# GM Info from Wiki

GM_NONSTANDARD_NAMES = {
    'piano':'Acoustic Grand Piano',
    'acoustic-piano': 'Acoustic Grand Piano',
    'grand-piano': 'Acoustic Grand Piano',
    'bright-piano': 'Bright Acoustic Piano',
    'electric-piano': 'Electric Piano 1',
    'elec-piano': 'Electric Piano 1',
}

GM_PROG_NAMES = [
        "Acoustic Grand Piano", #1
        "Bright Acoustic Piano", #2
        "Electric Grand Piano", #3
        "Honky-tonk Piano", #4
        "Electric Piano 1", #5
        "Electric Piano 2", #6
        "Harpsichord", #7
        "Clavinet", #8
        "Celesta", #9
        "Glockenspiel", #10
        "Music Box", #11
        "Vibraphone", #12
        "Marimba", #13
        "Xylophone", #14
        "Tubular Bells", #15
        "Dulcimer", #16
        "Drawbar Organ", #17
        "Percussive Organ", #18
        "Rock Organ", #19
        "Church Organ", #20
        "Reed Organ", #21
        "Accordion", #22
        "Harmonica", #23
        "Tango Accordion", #24
        "Acoustic Guitar (nylon)", #25
        "Acoustic Guitar (steel)", #26
        "Electric Guitar (jazz)", #27
        "Electric Guitar (clean)", #28
        "Electric Guitar (muted)", #29
        "Electric Guitar (overdriven)", #30
        "Electric Guitar (distortion)", #31
        "Electric Guitar (harmonics)", #32
        "Acoustic Bass", #33
        "Electric Bass (finger)", #34
        "Electric Bass (picked)", #35
        "Fretless Bass", #36
        "Slap Bass 1", #37
        "Slap Bass 2", #38
        "Synth Bass 1", #39
        "Synth Bass 2", #40
        "Violin", #41
        "Viola", #42
        "Cello", #43
        "Contrabass", #44
        "Tremolo Strings", #45
        "Pizzicato Strings", #46
        "Orchestral Harp", #47
        "Timpani", #48
        "String Ensemble 1", #49
        "String Ensemble 2", #50
        "Synth Strings 1", #51
        "Synth Strings 2", #52
        "Choir Aahs", #53
        "Voice Oohs (or Doos)", #54
        "Synth Voice or Solo Vox", #55
        "Orchestra Hit", #56
        "Trumpet", #57
        "Trombone", #58
        "Tuba", #59
        "Muted Trumpet", #60
        "French Horn", #61
        "Brass Section", #62
        "Synth Brass 1", #63
        "Synth Brass 2", #64
        "Soprano Sax", #65
        "Alto Sax", #66
        "Tenor Sax", #67
        "Baritone Sax", #68
        "Oboe", #69
        "English Horn", #70
        "Bassoon", #71
        "Clarinet", #72
        "Piccolo", #73
        "Flute", #74
        "Recorder", #75
        "Pan Flute", #76
        "Blown bottle", #77
        "Shakuhachi", #78
        "Whistle", #79
        "Ocarina", #80
        "Lead 1 (square)", #81
        "Lead 2 (sawtooth)", #82
        "Lead 3 (calliope)", #83
        "Lead 4 (chiff)", #84
        "Lead 5 (charang)", #85 guitar-like
        "Lead 6 (space voice)", #86
        "Lead 7 (fifths)", #87
        "Lead 8 (base and lead)", #88
        "Pad 1 (new age or fantasia)", #89 new age or fantasia, a warm pad stacked with a bell
        "Pad 2 (warm)", #90
        "Pad 3 (polysynth or poly)", #91
        "Pad 4 (choir)", #92
        "Pad 5 (bowed glass or bowed)", #93
        "Pad 6 (metallic)", #94
        "Pad 7 (halo)", #95
        "Pad 8 (sweep)", #96
        "FX 1 (rain)", #97
        "FX 2 (soundtrack)", #98
        "FX 3 (crystal)", #99
        "FX 4 (atmosphere)", #100
        "FX 5 (brightness)", #101
        "FX 6 (goblins)", #102
        "FX 7 (echoes)", #103
        "FX 8 (sci-fi)", #104
        "Sitar", #105
        "Banjo", #106
        "Shamisen", #107
        "Koto", #108
        "Kalimba", #109
        "Bag Pipe", #110
        "Fiddle", #111
        "Shanai", #112
        "Tinkle Bell", #113
        "Agogo", #114
        "Steel Drums", #115
        "Woodblock", #116
        "Taiko Drum", #117
        "Melodic Tom or 808 Toms", #118
        "Synth Drum", #119
        "Reverse Cymbal", #120
        "Guitar Fret Noise", #121
        "Breath Noise", #122
        "Seashore", #123
        "Bird Tweet", #124
        "Telephone Ring", #125
        "Helicopter", #126
        "Applause", #127
        "Gunshot", #128
]

NEAREST_PROGRAMS = {
    "Acoustic Grand Piano":range(0,8), #1 Piano
    "Bright Acoustic Piano":range(0,8), #2
    "Electric Grand Piano":range(0,8), #3
    "Honky-tonk Piano":range(0,8), #4
    "Electric Piano 1":range(0,8), #5
    "Electric Piano 2":range(0,8), #6
    "Harpsichord":range(0,8), #7
    "Clavinet":range(0,8), #8
    "Celesta":range(8,16), #9 Chromatic Percussion
    "Glockenspiel":range(8,16), #10
    "Music Box":range(8,16), #11
    "Vibraphone":range(8,16), #12
    "Marimba":range(8,16), #13
    "Xylophone":range(8,16), #14
    "Tubular Bells":range(8,16), #15
    "Dulcimer":range(8,16), #16
    "Drawbar Organ":range(16,24), #17 Organ
    "Percussive Organ":range(16,24), #18
    "Rock Organ":range(16,24), #19
    "Church Organ":range(16,24), #20
    "Reed Organ":range(16,24), #21
    "Accordion":range(16,24), #22
    "Harmonica":range(16,24), #23
    "Tango Accordion":range(16,24), #24
    "Acoustic Guitar (nylon)":range(24,32), #25 Guitar
    "Acoustic Guitar (steel)":range(24,32), #26
    "Electric Guitar (jazz)":range(24,32), #27
    "Electric Guitar (clean)":range(24,32), #28
    "Electric Guitar (muted)":range(24,32), #29
    "Electric Guitar (overdriven)":range(24,32), #30
    "Electric Guitar (distortion)":range(24,32), #31
    "Electric Guitar (harmonics)":range(24,32), #32
    "Acoustic Bass":range(32,40), #33 Bass
    "Electric Bass (finger)":range(32,40), #34
    "Electric Bass (picked)":range(32,40), #35
    "Fretless Bass":range(32,40), #36
    "Slap Bass 1":range(32,40), #37
    "Slap Bass 2":range(32,40), #38
    "Synth Bass 1":range(32,40), #39
    "Synth Bass 2":range(32,40), #40
    "Violin":range(40,48), #41 Strings
    "Viola":range(40,48), #42
    "Cello":range(40,48), #43
    "Contrabass":range(40,48), #44
    "Tremolo Strings":range(40,48), #45
    "Pizzicato Strings":range(40,48), #46
    "Orchestral Harp":range(40,48), #47
    "Timpani":range(40,48), #48
    "String Ensemble 1":range(48,56), #49 Ensemble
    "String Ensemble 2":range(48,56), #50
    "Synth Strings 1":range(48,56), #51
    "Synth Strings 2":range(48,56), #52
    "Choir Aahs":range(48,56), #53
    "Voice Oohs (or Doos)":range(48,56), #54
    "Synth Voice or Solo Vox":range(48,56), #55
    "Orchestra Hit":range(48,56), #56
    "Trumpet":range(56,64), #57 Brass
    "Trombone":range(56,64), #58
    "Tuba":range(56,64), #59
    "Muted Trumpet":range(56,64), #60
    "French Horn":range(56,64), #61
    "Brass Section":range(56,64), #62
    "Synth Brass 1":range(56,64), #63
    "Synth Brass 2":range(56,64), #64
    "Soprano Sax":range(64,72), #65 Reed
    "Alto Sax":range(64,72), #66
    "Tenor Sax":[66,65,67,64,68,69,70,71], #67
    "Baritone Sax":[67,66,65,64,68,69,70], #68
    "Oboe":range(64,72), #69
    "English Horn":range(64,72), #70
    "Bassoon":range(64,72), #71
    "Clarinet":range(64,72), #72
    "Piccolo":range(72,80), #73 Pipe
    "Flute":range(72,80), #74
    "Recorder":range(72,80), #75
    "Pan Flute":range(72,80), #76
    "Blown bottle":range(72,80), #77
    "Shakuhachi":range(72,80), #78
    "Whistle":range(72,80), #79
    "Ocarina":range(72,80), #80
    "Lead 1 (square)":range(80,88), #81 Synth Lead
    "Lead 2 (sawtooth)":range(80,88), #82
    "Lead 3 (calliope)":range(80,88), #83
    "Lead 4 (chiff)":range(80,88), #84
    "Lead 5 (charang)":range(80,88), #85 guitar-like
    "Lead 6 (space voice)":range(80,88), #86
    "Lead 7 (fifths)":range(80,88), #87
    "Lead 8 (base and lead)":range(80,88), #88
    "Pad 1 (new age or fantasia)":range(88,96), #89 Synth Pad
    "Pad 2 (warm)":range(88,96), #90
    "Pad 3 (polysynth or poly)":range(88,96), #91
    "Pad 4 (choir)":range(88,96), #92
    "Pad 5 (bowed glass or bowed)":range(88,96), #93
    "Pad 6 (metallic)":range(88,96), #94
    "Pad 7 (halo)":range(88,96), #95
    "Pad 8 (sweep)":range(88,96), #96
    "FX 1 (rain)":range(96,104), #97 Synth Effects
    "FX 2 (soundtrack)":range(96,104), #98
    "FX 3 (crystal)":range(96,104), #99
    "FX 4 (atmosphere)":range(96,104), #100
    "FX 5 (brightness)":range(96,104), #101
    "FX 6 (goblins)":range(96,104), #102
    "FX 7 (echoes)":range(96,104), #103
    "FX 8 (sci-fi)":range(96,104), #104
    "Sitar":range(104,112), #105 Ethnic
    "Banjo":range(104,112), #106
    "Shamisen":range(104,112), #107
    "Koto":range(104,112), #108
    "Kalimba":range(104,112), #109
    "Bag Pipe":range(104,112), #110
    "Fiddle":range(104,112), #111
    "Shanai":range(104,112), #112
    "Tinkle Bell":range(112,120), #113 Percussive
    "Agogo":range(112,120), #114
    "Steel Drums":range(112,120), #115
    "Woodblock":range(112,120), #116
    "Taiko Drum":range(112,120), #117
    "Melodic Tom or 808 Toms":range(112,120), #118
    "Synth Drum":range(112,120), #119
    "Reverse Cymbal":range(112,120), #120
    "Guitar Fret Noise":range(120,128), #121 Sound Effects
    "Breath Noise":range(120,128), #122
    "Seashore":range(120,128), #123
    "Bird Tweet":range(120,128), #124
    "Telephone Ring":range(120,128), #125
    "Helicopter":range(120,128), #126
    "Applause":range(120,128), #127
    "Gunshot":range(120,128), #128
}

GM_NONSTANDARD_PERCUSSIVE_NAMES = {
    'bass-drum':'Acoustic Bass Drum'
}

GM_PERCUSSIVE_NAMES = [
        "High Q", #27 (Roland Extension)
        "Slap", #28 (Roland Extension)
        "Scratch Push", #29 (Roland Extension)
        "Scratch Pull", #30 (Roland Extension)
        "Sticks", #31 (Roland Extension)
        "Square Click", #32 (Roland Extension)
        "Metronome Click", #33 (Roland Extension)
        "Metronome Bell", #34 (Roland Extension)
        "Acoustic Bass Drum", #35
        "Electric Bass Drum", #36
        "Side Stick", #37
        "Acoustic Snare", #38
        "Hand Clap", #39
        "Electric Snare", #40
        "Low Floor Tom", #41
        "Closed Hi-hat", #42
        "High Floor Tom", #43
        "Pedal Hi-hat", #44
        "Low Tom", #45
        "Open Hi-hat", #46
        "Low-Mid Tom", #47
        "Hi-Mid Tom", #48
        "Crash Cymbal 1", #49
        "High Tom", #50
        "Ride Cymbal 1", #51
        "Chinese Cymbal", #52
        "Ride Bell", #53
        "Tambourine", #54
        "Splash Cymbal", #55
        "Cowbell", #56
        "Crash Cymbal 2", #57
        "Vibraslap", #58
        "Ride Cymbal 2", #59
        "High Bongo", #60
        "Low Bongo", #61
        "Mute High Conga", #62
        "Open High Conga", #63
        "Low Conga", #64
        "High Timbale", #65
        "Low Timbale", #66
        "High Agogo", #67
        "Low Agogo", #68
        "Cabasa", #69
        "Maracas", #70
        "Short Whistle", #71
        "Long Whistle", #72
        "Short Guiro", #73
        "Long Guiro", #74
        "Claves", #75
        "High Woodblock", #76
        "Low Woodblock", #77
        "Mute Cuica", #78
        "Open Cuica", #79
        "Mute Triangle", #80
        "Open Triangle", #81
        "Shaker", #82 (Roland Extension)
        "Jingle Bell", #83 (Roland Extension)
        "Belltree" , #84 (Roland Extension)
        "Castanets", #85 (Roland Extension)
        "Mute Surdo", #86 (Roland Extension)
        "Open Surdo", #87 (Roland Extension)
    ]

# Roland kit information from:
# https://www.voidaudio.net/percussion.html

ROLAND_ROOM_KIT = [x for x in GM_PERCUSSIVE_NAMES]
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("Acoustic Snare")]="Snare Drum 1"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("Electric Snare")]="Snare Drum 2"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("Low Floor Tom")]="Room Low Tom 2"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("High Floor Tom")]="Room Low Tom 1"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("Low Tom")]="Room Mid Tom 2"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("Low-Mid Tom")]="Room Mid Tom 1"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("Hi-Mid Tom")]="Room High Tom 1"
ROLAND_ROOM_KIT[GM_PERCUSSIVE_NAMES.index("High Tom")]="Room High Tom 1"

ROLAND_POWER_KIT = [x for x in ROLAND_ROOM_KIT]
ROLAND_POWER_KIT[ROLAND_ROOM_KIT.index("Electric Bass Drum")]="Mondo Kick"
ROLAND_POWER_KIT[ROLAND_ROOM_KIT.index("Snare Drum 1")]="Gated SD"

ROLAND_ELECTRONIC_KIT = [x for x in ROLAND_ROOM_KIT]
ROLAND_ELECTRONIC_KIT[ROLAND_ROOM_KIT.index("Electric Bass Drum")]="Elec BD"
ROLAND_ELECTRONIC_KIT[ROLAND_ROOM_KIT.index("Snare Drum 1")]="Elec BD"
ROLAND_ELECTRONIC_KIT[ROLAND_ROOM_KIT.index("Snare Drum 2")]="Gated SD"
ROLAND_ELECTRONIC_KIT[GM_PERCUSSIVE_NAMES.index("Low Floor Tom")]="Elec Low Tom 2"
ROLAND_ELECTRONIC_KIT[GM_PERCUSSIVE_NAMES.index("High Floor Tom")]="Elec Low Tom 1"
ROLAND_ELECTRONIC_KIT[GM_PERCUSSIVE_NAMES.index("Low Tom")]="Elec Mid Tom 2"
ROLAND_ELECTRONIC_KIT[GM_PERCUSSIVE_NAMES.index("Low-Mid Tom")]="Elec Mid Tom 1"
ROLAND_ELECTRONIC_KIT[GM_PERCUSSIVE_NAMES.index("Hi-Mid Tom")]="Elec High Tom 1"
ROLAND_ELECTRONIC_KIT[GM_PERCUSSIVE_NAMES.index("High Tom")]="Elec High Tom 1"

ROLAND_TR808_KIT = [x for x in ROLAND_ROOM_KIT]
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Electric Bass Drum")]="808 Bass Drum"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Side Stick")]="808 Rim Shot"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Snare Drum 1")]="808 Snare Drum"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Snare Drum 2")]="Snare Drum 2"
ROLAND_TR808_KIT[GM_PERCUSSIVE_NAMES.index("Low Floor Tom")]="808 Low Tom 2"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Closed Hi-hat")]="808 Closed Hi-hat"
ROLAND_TR808_KIT[GM_PERCUSSIVE_NAMES.index("High Floor Tom")]="808 Low Tom 1"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Pedal Hi-hat")]="808 Closed Hi-hat"
ROLAND_TR808_KIT[GM_PERCUSSIVE_NAMES.index("Low Tom")]="808 Mid Tom 2"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Open Hi-hat")]="808 Closed Hi-hat"
ROLAND_TR808_KIT[GM_PERCUSSIVE_NAMES.index("Low-Mid Tom")]="808 Mid Tom 1"
ROLAND_TR808_KIT[GM_PERCUSSIVE_NAMES.index("Hi-Mid Tom")]="808 High Tom 1"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Crash Cymbal 1")]="808 Cymbal"
ROLAND_TR808_KIT[GM_PERCUSSIVE_NAMES.index("High Tom")]="808 High Tom 1"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Cowbell")]="808 Cowbell"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Mute High Conga")]="808 High Conga"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Open High Conga")]="808 Mid Conga"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Low Conga")]="808 Low Conga"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Maracas")]="808 Maracas"
ROLAND_TR808_KIT[ROLAND_ROOM_KIT.index("Claves")]="808 Claves"

ROLAND_JAZZ_KIT = [x for x in GM_PERCUSSIVE_NAMES]
ROLAND_JAZZ_KIT[GM_PERCUSSIVE_NAMES.index("Acoustic Bass Drum")]="Jazz BD 2"
ROLAND_JAZZ_KIT[GM_PERCUSSIVE_NAMES.index("Electric Bass Drum")]="Jazz BD 1"
ROLAND_JAZZ_KIT[GM_PERCUSSIVE_NAMES.index("Acoustic Snare")]="Snare Drum 1"
ROLAND_JAZZ_KIT[GM_PERCUSSIVE_NAMES.index("Electric Snare")]="Snare Drum 2"

ROLAND_BRUSH_KIT = [x for x in ROLAND_JAZZ_KIT]
ROLAND_BRUSH_KIT[ROLAND_JAZZ_KIT.index("Snare Drum 1")]="Brush Tap"
ROLAND_BRUSH_KIT[ROLAND_JAZZ_KIT.index("Hand Clap")]="Brush Slap"
ROLAND_BRUSH_KIT[ROLAND_JAZZ_KIT.index("Snare Drum 2")]="Brush Swirl"




ROLAND_KITS = {
    0:{
        'name':"General Midi Level 1",
        'percussive_names':GM_PERCUSSIVE_NAMES,
    },
    8:{
        'name':"Room Kit",
        'percussive_names':ROLAND_ROOM_KIT
    },
    16:{
        'name':"Power Kit",
        'percussive_names':ROLAND_POWER_KIT
    },
    24:{
        'name':"Electronic Kit",
        'percussive_names':ROLAND_ELECTRONIC_KIT
    },
    25:{
        'name':"TR-808 Kit",
        'percussive_names':ROLAND_TR808_KIT
    },
    32:{
        'name':"Jazz Kit",
        'percussive_names':ROLAND_JAZZ_KIT
    },
    40:{
        'name':"Brush Kit",
        'percussive_names':ROLAND_BRUSH_KIT
    }
}

# Mappings created manually from inspecting sample files

PHILHARMONIA_MAP = {
    # instruments
    'banjo':[('Banjo',None,'normal')],
    'bassoon':[('Bassoon',None,'normal')],
    'cello':[('Cello',None,'arco-normal')],
    'clarinet':[('Clarinet',None,'normal')],
    'contrabassoon':[('Contrabass',None,'normal')],
    'english-horn':[('English Horn',None,'normal')],
    'double-bass':[('Fretless Bass',None,'arco-normal')],
    'flute':[('Flute',None,'normal')],
    'french-horn':[('French Horn',None,'normal')],
    'guitar':[('Electric Guitar (clean)',None,'normal'),('Electric Guitar (harmonics)',None,'harmonics')],
    'oboe':[('Oboe',None,'normal')],
    'saxophone':[('Alto Sax',None,'normal')],
    'trombone':[('Trombone',None,'normal')],
    'trumpet':[('Trumpet',None,'normal')],
    'tuba':[('Tuba',None,'normal')],
    'viola':[('Viola','1','arco-normal')], # viola very long has two-note sequences :-(
    'violin':[('Violin',None,'arco-normal')],
    # percussive
    'bass-drum':[('Acoustic Bass Drum',None,'struck-singly')],
    'cabasa':[('Cabasa',None,'effect')],
    'chinese-cymbal':[('Chinese Cymbal',None,'damped')],
    #'clash-cymbals':[('Crash Cymbal 1',None,'struck-together')],
    'cowbell':[('Cowbell',None,'undamped')],
    'snare-drum':[('Acoustic Snare','025','with-snares')],
    'tambourine':[('Tambourine',None,'hand')],
    'tom-toms':[('Low Floor Tom',None,'struck-singly')],
    'suspended-cymbal':[('Ride Cymbal 2',None,'vibe-mallet-undamped')],
}

PHILHARMONIA_DURATIONS = ['very-long','long',"1","15"]


UIOWA_MAP = {
    'piano':[('Acoustic Grand Piano',None)],
    'xylophone':[('Xylophone','hardrubber')],
    'bells':[('Tubular Bells','brass')],
    'vibraphone':[('Vibraphone','sustain')],
    'marimba':[('Marimba','rubber')],
    'sopsax':[('Soprano Sax','nonvib')]
}

UIOWA_MAP_PERCUSSION = {
    '21ride':[('Ride Cymbal 1','normal')],
    '13crash':[('Crash Cymbal 1','normal')],
    '17crash':[('Crash Cymbal 2','normal')],
    'hihat':[('Pedal Hi-hat','footsplash'),('Open Hi-hat','normal'),('Closed Hi-hat','footclose')],
}

BANK_METADATA = {
    'bankidx':0,
    'progidx':0,
    'note':0,
    'velocity':"",
    'duration':"",
    'sample_file':"",
    'description':""
}

KIT_METADATA = {
    'kitidx':0,
    'note':0,
    'velocity':"",
    'duration':"",
    'sample_file':"",
    'description':""
}


def ffmpegConvert(inputfile:str,sample_rate:int = AV_RATE):
    outputfile=f"{os.path.splitext(inputfile)[0]}.wav"
    return subprocess.call(['ffmpeg','-loglevel','quiet','-y', '-i', inputfile,'-ar',str(int(sample_rate)), outputfile])

class MIDISupport():
    def __init__(self,
                 mididir:str="./",
                 segment_thresh:float = 0.05,
                 silence_thresh:float = 0.02,
                 sample_rate:int = AV_RATE,
                 chunk_size:int = 1024,
                 dtype=np.float32):
        self.mididir = mididir
        self.banks = {}
        self.kits = {}
        self.silence_thresh = silence_thresh
        self.segment_thresh = segment_thresh
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.dtype=dtype
        self.segment_cache={}
        self.model_data_cache={}
        if len(mididir)>0:
            self.check_dir(mididir)
            self.prepare_bankdir()
            self.prepare_kitdir()
            self.scan_mididir()
        

    def read_sample_meta(self,
                         bankidx:int=None,
                         kitidx:int=None,
                         progidx:int=None,
                         notenum:int=None,
                         velocity:str=None):
        if bankidx != None:
            metafile = os.path.join(self.mididir,
                                    f"bank-{bankidx+1}",
                                    f"prog-{progidx+1}",
                                    f"note-{notenum}",
                                    f"velocity-{velocity}",
                                    f"meta.json")
        elif kitidx != None:
            metafile = os.path.join(self.mididir,
                                    f"kit-{kitidx+1}",
                                    f"note-{notenum}",
                                    f"velocity-{velocity}",
                                    f"meta.json")
        with open(metafile,"r") as file:
                metadata = json.load(file)
                return metadata
     
    def write_sample_meta(self,
                         metadata:dict,
                         bankidx:int=None,
                         kitidx:int=None,
                         progidx:int=None,
                         notenum:int=None,
                         velocity:str=None,
                         overwrite:bool=False):
        if bankidx != None:
            metafile = os.path.join(self.mididir,
                                    f"bank-{bankidx+1}",
                                    f"prog-{progidx+1}",
                                    f"note-{notenum}",
                                    f"velocity-{velocity}",
                                    f"meta.json")
        elif kitidx != None:
            metafile = os.path.join(self.mididir,
                                    f"kit-{kitidx+1}",
                                    f"note-{notenum}",
                                    f"velocity-{velocity}",
                                    f"meta.json")
        if os.path.exists(metafile) and not overwrite:
            return
        with open(metafile,"w") as file:
            json.dump(metadata,file,indent="\t",sort_keys=True)   
    
        
    def scan_mididir(self):
        """ Scan the midi dir for banks and kits and read all meta data. """

        self.banks = {}
        self.kits = {}
        for filename in os.listdir(self.mididir):
            if filename.startswith("bank"):
                _,bankidx = filename.split("-")
                bankidx=int(bankidx)-1
                self.banks[bankidx]={}
                for progidx in range(128):
                    self.banks[bankidx][progidx]={}
                    for note in range(128):
                        self.banks[bankidx][progidx][note]={}
                        for velocity in VELOCITY_SHORT.keys():
                            metadata = self.read_sample_meta(bankidx=bankidx,
                                                            progidx=progidx,
                                                            notenum=note,
                                                            velocity=velocity)
                            self.banks[bankidx][progidx][note][velocity]=metadata
            elif filename.startswith("kit"):
                _,kitidx= filename.split("-")
                kitidx=int(kitidx)-1
                self.kits[kitidx]={}
                notes = ROLAND_KITS[kitidx]['percussive_names']
                for noteidx in range(len(notes)):
                    note_num=noteidx+27
                    self.kits[kitidx][note_num]={}
                    for velocity in VELOCITY_SHORT.keys():
                        metadata = self.read_sample_meta(kitidx=kitidx,
                                                        notenum=note_num,
                                                        velocity=velocity)
                        self.kits[kitidx][note_num][velocity]=metadata


    def check_dir(self,dirname):
        """ Check and create a directory if needed. """

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
    def write_bank_name(self,
                        bank:int=0,
                        name:str="unknown",
                        overwrite:bool=False):
        file = os.path.join(self.mididir,f"bank-{bank+1}",
                            f"bank-name-{name.lower().replace(' ','_')}.txt")
        if not os.path.exists(file) or overwrite:
            with open(file,"w") as f:
                f.writelines([name])
    
    def write_program_name(self,
                           bank:int=0,
                           program:int=0,
                           name:str="unknown",
                           overwrite:bool=False):
        file = os.path.join(self.mididir,f"bank-{bank+1}",
                            f"prog-{program+1}",
                            f"program-name-{name.lower().replace(' ','_')}.txt")
        if not os.path.exists(file) or overwrite:
            with open(file,"w") as f:
                f.writelines([name])
            
    def write_note_name(self,
                        bank:int=0,
                        program:int=0,
                        note:int=0,
                        name:str="unknown",
                        overwrite:bool=False):
        file = os.path.join(self.mididir,f"bank-{bank+1}",
                            f"prog-{program+1}",
                            f"note-{note}",
                            f"note-name-{name.lower().replace(' ','_')}.txt")
        if not os.path.exists(file) or overwrite:
            with open(file,"w") as f:
                f.writelines([name])
    
    def write_percussive_name(self,
                              kit:int=0,
                              note:int=0,
                              name:str="unknown",
                              overwrite:bool=False):
        file = os.path.join(self.mididir,f"kit-{kit+1}",
                            f"note-{note}",
                            f"percussive-name-{name.lower().replace(' ','_')}.txt")
        if not os.path.exists(file) or overwrite:
            with open(file,"w") as f:
                f.writelines([name])
                
    def write_kit_name(self,
                       kit:int=0,
                       name:str="unknown",
                       overwrite:bool=False):
        file = os.path.join(self.mididir,f"kit-{kit+1}",
                            f"kit-name-{name.lower().replace(' ','_')}.txt")
        if not os.path.exists(file) or overwrite:
            with open(file,"w") as f:
                f.writelines([name])

    def prepare_bankdir(self):
        """ Create the default bank directory structure and populate with default programs. """
        
        bank1dir = os.path.join(self.mididir,"bank-1")
        self.check_dir(bank1dir)
        self.write_bank_name(bank=0,name="General Midi Level 1")
        
        for progidx in range(128):
            programdir = os.path.join(bank1dir,f"prog-{progidx+1}")
            self.check_dir(programdir)
            self.write_program_name(bank=0,program=progidx,name=GM_PROG_NAMES[progidx])
            for note_num in range(128):
                notedir = os.path.join(programdir,f"note-{note_num}")
                self.check_dir(notedir)
                self.write_note_name(bank=0,program=progidx,note=note_num,name=NOTE_NAMES[note_num])
                for velocity in VELOCITY_SHORT.keys():
                    velocitydir = os.path.join(notedir,f"velocity-{velocity}")
                    self.check_dir(velocitydir)
                    self.write_sample_meta({
                            'bankidx':0,
                            'progidx':progidx,
                            'note':note_num,
                            'velocity':velocity,
                            'duration':None,
                            'sample_file':None,
                            'description':None
                        },bankidx=0,progidx=progidx,notenum=note_num,velocity=velocity)
                    
    def prepare_kitdir(self):
        """ Create the default kit directory structure and populate with default programs. """
        
        for kitidx in ROLAND_KITS.keys():
            kitdir = os.path.join(self.mididir,f"kit-{kitidx+1}")
            self.check_dir(kitdir)
            self.write_kit_name(kitidx,ROLAND_KITS[kitidx]['name'])
            notes = ROLAND_KITS[kitidx]['percussive_names']
            for noteidx in range(len(notes)):
                note_num = noteidx+27
                notedir = os.path.join(kitdir,f"note-{note_num}")
                self.check_dir(notedir)
                self.write_percussive_name(kit=kitidx,note=note_num,name=notes[noteidx])
                for velocity in VELOCITY_SHORT.keys():
                        velocitydir = os.path.join(notedir,f"velocity-{velocity}")
                        self.check_dir(velocitydir)
                        self.write_sample_meta({
                                'kitidx':kitidx,
                                'note':note_num,
                                'velocity':velocity,
                                'duration':None,
                                'sample_file':None,
                                'description':None
                            },kitidx=kitidx,notenum=note_num,velocity=velocity)

    
    def parse_note_array(self,note:str):
        note = note.lower().replace("#","s")
        if note.capitalize() in NOTE_NUMS:
            return [NOTE_NUMS[note.capitalize()]]
        if note.capitalize() in NOTE_NUMS_ALT:
            return [NOTE_NUMS_ALT[note.capitalize()]]
        for i in [2,3,4]:
            if i>=len(note):
                return None
            note1=note[0:i].capitalize()
            if note1 in NOTE_NUMS:
                note2=note[i:].capitalize()
                if note2 in NOTE_NUMS:
                    return [NOTE_NUMS[note1],
                            NOTE_NUMS[note2]]
                if note2 in NOTE_NUMS_ALT:
                    return [NOTE_NUMS[note1],
                            NOTE_NUMS_ALT[note2]]
            if note1 in NOTE_NUMS_ALT:
                note2=note[i:].capitalize()
                if note2 in NOTE_NUMS:
                    return [NOTE_NUMS_ALT[note1],
                            NOTE_NUMS[note2]]
                if note2 in NOTE_NUMS_ALT:
                    return [NOTE_NUMS_ALT[note1],
                            NOTE_NUMS_ALT[note2]]
        return None


    def determine_sample_mapping(self,filename:str):
        """ Given a filename to a sample, determine where it should be mapped to. """
        
        basename = filename.replace(".wav","").lower()
        parts = [x.lower() for x in basename.split("_")]
        parts2 = [x.lower() for x in basename.split(".")]
        
        # University of Iowa Electronic Music Studios
        # https://theremin.music.uiowa.edu/MIS.html
        #
        # instrument.nuance.notename
        # instrument.special.nuance.notename
        # instrument.special.nuance.noterange [noterange = ABXY where B and Y are octave numbers]
        #
        # notenames use only flat (b) notation

        if parts2[0] in UIOWA_MAP and len(parts2)==3:
            instrument, nuance, note = parts2
            maps = UIOWA_MAP[instrument]
            for map,special in maps:
                if map in GM_PROG_NAMES:
                    notenum = self.parse_note_array(note)
                    if notenum != None:
                        if nuance in VELOCITY_SHORT:
                            return True,GM_PROG_NAMES.index(map),\
                                notenum,nuance,\
                                "very-long","University of Iowa, Electronic Music Studios"
        if parts2[0] in UIOWA_MAP and (len(parts2)==4 or (len(parts2)==5 and parts2[4]=="stereo")):
            if len(parts2)==4:
                instrument, special, nuance, note = parts2
            else:
                instrument, special, nuance, note, _ = parts2
            maps = UIOWA_MAP[instrument]
            for map,special2 in maps:
                if special==special2:
                    if nuance in VELOCITY_SHORT:
                        notenum = self.parse_note_array(note)
                        if notenum != None:
                            return True,GM_PROG_NAMES.index(map),\
                                notenum,nuance,\
                                    "very-long","University of Iowa, Electronic Music Studios"
        if parts2[0] in UIOWA_MAP_PERCUSSION and (len(parts2)==3 or len(parts2)==4):
            if len(parts2)==3:
                instrument, special, nuance = parts2
            else:
                instrument, stick, special, nuance = parts2
            maps = UIOWA_MAP_PERCUSSION[instrument]
            for map,special2 in maps:
                if special==special2:
                    if nuance in VELOCITY_SHORT:
                        notenum = [GM_PERCUSSIVE_NAMES.index(map)+27]
                        return False,None,\
                                notenum,\
                                    nuance,\
                                        "very long",\
                                            "University of Iowa, Electronic Music Studios"

        # http://sonimusicae.free.fr/Banques/SoniMusicae-Blanchet1720-sf2.zip

        # Philharmonia Orchestra
        # https://philharmonia.co.uk/resources/sound-samples/
        #
        # instrument_notename_duration_nuance_special
        #
        # note names use only sharp (s) notation

        
        if len(parts) == 5:
            instrument, note, duration, nuance, special = parts
            if instrument in PHILHARMONIA_MAP:
                maps = PHILHARMONIA_MAP[instrument]
                for map, required_duration, required_special in maps:
                    print(map,required_duration,required_special)
                    if map in GM_PERCUSSIVE_NAMES:
                        if note == "":
                            if (required_duration!=None and duration==required_duration) or\
                                 (required_duration==None and duration in PHILHARMONIA_DURATIONS):
                                if nuance in VELOCITY.values():
                                    if required_special==special:
                                        return False,None,\
                                            [GM_PERCUSSIVE_NAMES.index(map)+27],\
                                                VELOCITY_CONV[nuance],duration,\
                                                    "Philharmonia Orchestra"
                    elif note.capitalize() in NOTE_NUMS:
                        if (required_duration!=None and duration==required_duration) or\
                                 (required_duration==None and duration in PHILHARMONIA_DURATIONS):
                            if nuance in VELOCITY.values():
                                if required_special==special:
                                    return True,\
                                        GM_PROG_NAMES.index(map),\
                                            [NOTE_NUMS[note.capitalize()]],\
                                                VELOCITY_CONV[nuance],duration,"Philharmonia Orchestra"

        # instrument_velocity_note
        # instrument.velocity.note
        # instrument_note_velocity
        # instrument.note.velocity
        # instrument.velocity
        # instrument_velocity
        for parts3 in  [parts, parts2]:
            progidx=None
            kitidx=None
            notenum=None
            duration=None
            if len(parts3) == 2:
                for instrument in [parts3[0], 
                                parts3[0].replace("-"," "),
                                parts3[0].replace(" ","-")]:
                    if instrument not in GM_PERCUSSIVE_NAMES:
                        if instrument in GM_NONSTANDARD_PERCUSSIVE_NAMES:
                            notenum = GM_PERCUSSIVE_NAMES.index(GM_NONSTANDARD_PERCUSSIVE_NAMES[instrument])+27
                        else:
                            continue
                    else:
                        notenum = GM_PERCUSSIVE_NAMES.index(instrument)+27
                    if parts[1] in VELOCITY_SHORT:
                        return False,\
                                progidx,\
                                [notenum],\
                                velocity,\
                                "very long",\
                                "unknown origin"
            elif len(parts3) == 3:
                for instrument in [parts3[0], 
                                parts3[0].replace("-"," "),
                                parts3[0].replace(" ","-")]:
                    if instrument not in GM_PROG_NAMES:
                        if instrument in GM_NONSTANDARD_NAMES:
                            progidx = GM_PROG_NAMES.index(GM_NONSTANDARD_NAMES[instrument])
                        else:
                            continue
                    else:
                        progidx = GM_PROG_NAMES.index(instrument)
                    if parts3[1] not in VELOCITY_SHORT:
                        if parts3[2] not in VELOCITY_SHORT:
                            continue
                        else:
                            velocity = parts3[2]
                            notenum = self.parse_note_array(parts3[1])
                    else:
                        velocity = parts3[1]
                        notenum = self.parse_note_array(parts3[2])
                    if notenum != None:
                        return True,\
                                progidx,\
                                notenum,\
                                velocity,\
                                "very long",\
                                "unknown origin"

        return None

    def decode(self,in_data, signal_desc, norm):
        result = np.frombuffer(in_data, dtype=sw_dtype(signal_desc.sampwidth))
        chunk_length = len(result) // signal_desc.nchannels
        result = np.reshape(result, (chunk_length, signal_desc.nchannels))
        return result/norm

    def process_sample(self,data,velocity,sample_idx,number_notes,big_segments=None):
        """ Remove silence, normalize velocity and split individual notes from the sample. """

        diff_len:int=10
        min_seg_len:int=100
        silence_seg_len:int=100

        velocity_norm = float(VELOCITY_SHORT[velocity])/127.0
        data = data - np.mean(data)
        amplitude = np.abs(data)
        max_amplitude = np.max(amplitude)
        data /= max_amplitude
        diff = np.abs(data[diff_len:-1]-data[0:-1-diff_len])
        diff = np.convolve(diff[:,0],np.ones(diff_len))[diff_len-1:-(diff_len-1)]
        max_diff = np.max(diff)
        
        
        if big_segments==None:
            segments=np.arange(len(diff))[diff > max_diff*self.segment_thresh]
            i=1
            big_segments=[]
            current_segment_start=0
            gap_threshold = len(data)/(4*number_notes)
            # find generally where the notes are being played
            while i<len(segments):
                # dist between this sample and the last
                dist = segments[i] - segments[i-1]
                seg_len = segments[i-1] - segments[current_segment_start] 
                if dist>gap_threshold or seg_len > len(data)/number_notes:
                    if seg_len>min_seg_len:
                        big_segments.append((segments[current_segment_start],segments[i-1]))
                    current_segment_start=i
                i+=1
            if len(big_segments)<number_notes:
                big_segments.append((segments[current_segment_start],segments[i-1]))
                current_segment_start=i
            #print("initial segments",big_segments)
            # refine the start of attack time, based on silence thresholding
            attack_segment_start=[]
            for segment in big_segments:
                i = segment[0]
                hit=False
                while i-silence_seg_len>=0:
                    maxm = np.max(data[i-silence_seg_len:i])
                    minm = np.min(data[i-silence_seg_len:i])
                    #print(maxm,minm,maxm-minm)
                    if (maxm-minm < self.silence_thresh):
                        attack_segment_start.append((i,segment[1]))
                        hit=True
                        break
                    i-=silence_seg_len//2
                if not hit:
                    attack_segment_start.append((0,segment[1]))
            # the note sample is now from the start of attack,
            # to just before the next start of attack
            big_segments=[]
            for i in range(len(attack_segment_start)-1):
                big_segments.append((attack_segment_start[i][0],attack_segment_start[i+1][0]-silence_seg_len))
            big_segments.append((attack_segment_start[-1][0],len(data)-1))

            if len(big_segments) != number_notes:
                if len(big_segments) < number_notes:
                    print(big_segments)
                    print([x[1]-x[0] for x in big_segments],[(x[1]-x[0])/len(data) for x in big_segments])
                    raise Exception(f"Number of segments {len(big_segments)} is less than what is required {number_notes}.")
                else:
                    print(big_segments)
                    print([x[1]-x[0] for x in big_segments],[(x[1]-x[0])/len(data) for x in big_segments])
                    raise Exception(f"Number of segments {len(big_segments)} is greater than what is required {number_notes}.")
            #print("final segments",big_segments)  
        data *= velocity_norm
        return data[big_segments[sample_idx][0]:big_segments[sample_idx][1]+1],big_segments        
        

    def read_sample(self,
                    filename:str,
                    resample:bool=True,
                    normalize:bool=True):
        sample_rate = self.sample_rate
        chunk_size = self.chunk_size
        dtype = self.dtype
        
        try:
            wf=wave.open(filename, 'rb')
            signal_desc = wf.getparams()
            model_data=np.empty(shape=(0,1),dtype=dtype)
            norm = 2**(signal_desc.sampwidth*8-1) if normalize else 1.0
            conversion = sample_rate/signal_desc.framerate
            if conversion > 1.0 and resample:  # need to upsample
                print("upsampling by factor",conversion)
                resample_module = Resample(conversion,sample_rate=sample_rate)
                lpf_module = IIRDesign(wp=[sample_rate/2.0*0.999],
                                            ws=[sample_rate/2.0],
                                            sample_rate=sample_rate,
                                            polled=True)
                resample_module.connect(lpf_module)
            elif conversion < 1.0 and resample:  # need to downsample
                print("downsampling by factor",conversion)
                resample_module = Resample(factor=conversion,
                                                sample_rate=sample_rate,
                                                polled=True)
                lpf_module = IIRDesign(wp=[sample_rate/2.0*0.999],
                                            ws=[sample_rate/2.0],
                                            sample_rate=sample_rate)
                lpf_module.connect(resample_module)
            if conversion > 1.0 and resample:
                data_chunks=[]
                while True:
                    data = wf.readframes(chunk_size)
                    
                    if data != '':
                        npdata = self.decode(data, signal_desc, norm).astype(dtype)
                        if len(npdata)==0:
                            break
                        data_chunks.append(npdata)
                        
                        continue
                    else:
                        break
                resample_module.receive_signal(np.concatenate(data_chunks))
                resample_module.process_all()
                lpf_module.process_all()
                data = lpf_module.get_out_buf().get_all()
                if data.shape[1]==1:
                    model_data = data[:,[0]]
                else: 
                    model_data = data[:,[0]] #0.5*(data[:,[0]]+data[:,[1]])
            elif conversion < 1.0 and resample:
                data_chunks=[]
                while True:
                    data = wf.readframes(chunk_size)
                    if data != '':
                        npdata = self.decode(data, signal_desc, norm).astype(dtype)
                        if len(npdata)==0:
                            break
                        data_chunks.append(npdata)
                        
                        continue
                    else:
                        break
                lpf_module.receive_signal(np.concatenate(data_chunks))
                lpf_module.process_all()
                resample_module.proces_all()
                data = resample_module.get_out_buf().get_all()
                if data.shape[1]==1:
                    model_data = data[:,[0]]
                else: 
                    model_data = data[:,[0]] #0.5*(data[:,[0]]+data[:,[1]])
            else:
                data_chunks=[]
                while True:
                    data = wf.readframes(chunk_size)
                    if data != '':
                        npdata = self.decode(data, signal_desc, norm).astype(dtype)
                        if len(npdata)==0:
                            break
                        if npdata.shape[1]==1:
                            data_chunks.append(npdata[:,[0]])
                        else:
                            data_chunks.append(npdata[:,[0]])
                        continue
                    else:
                        break
                model_data=np.concatenate(data_chunks)
            wf.close()
            return model_data
        except Exception as e:
            print(e)
            return None
    
    def write_model_data(self,filename:str,data:np.ndarray):
        """ Write WAV data to the file. """
        
        towavefile = ToWavFile(filename,
                               data.shape[1],
                               sample_rate=self.sample_rate,
                               chunk_size=self.chunk_size,
                               dtype=self.dtype)
        towavefile.open()
        towavefile.receive_signal(data)
        while towavefile.process_next_chunk() == AM_CONTINUE:
            pass
        towavefile.close()
    
    def import_sample(self,
                      filename:str,
                      metadata:dict,
                      duration:str,
                      velocity:str,
                      description:str,
                      modeldir:str,
                      sample_idx:int,
                      number_samples:int):
        data = self.read_sample(filename,resample=True)
        metadata["duration"] = duration
        metadata["sample_file"] = os.path.basename(filename)
        metadata["velocity"] = velocity
        metadata["description"] = f"{description}. Imported {time.asctime()}."
        modelfile = os.path.join(modeldir,metadata["sample_file"])
        if filename in self.segment_cache:
            big_segments = self.segment_cache[filename]
        else:
            big_segments = None
        print(f"Processing {filename}")
        data,big_segments=self.process_sample(data,velocity,sample_idx,number_samples,big_segments=big_segments)
        self.segment_cache[filename]=big_segments

        self.write_model_data(modelfile,data)
        print(f"{os.path.basename(filename)} -> {modelfile}")
    
    def import_samples(self,
                        dirname:str="./",
                        bankidx:int=0,
                        kitidx:int=0,
                        overwrite:bool=False,
                        create_backups:bool=False):
        """ Import wav file samples found in the directory (recursively scanned), copying them to the midi dir.
        
        Intelligent recognition of samples is attempted. The midi directory must be scanned prior to calling
        this method, using `scan_mididir()`.
        """

        self.segment_cache={}
        for filename in os.listdir(dirname):
            if os.path.isdir(os.path.join(dirname,filename)):
                self.import_samples(os.path.join(dirname,filename),
                                    bankidx=bankidx,
                                    kitidx=kitidx,
                                    overwrite=overwrite)
            else:
                filepath = os.path.join(dirname,filename)
                if filename.startswith("SYNTHED"):
                    print(f"{os.path.basename(filename)} starts with SYNTHED which is reserved. Rename the file and try again -> discarding")
                    continue
                if filename.lower().endswith((".aiff",".aif",".mp3")) and not os.path.exists(f"{os.path.splitext(filepath)[0]}.wav"):
                    ret=ffmpegConvert(os.path.join(dirname,filename))
                    if ret!=0:
                        print(f"{os.path.basename(filename)} not converted to wav file -> discarding")
                        continue
                    print(f"{os.path.basename(filename)} converted to wav file")
                    filename=f"{os.path.splitext(filename)[0]}.wav"
                if filename.lower().endswith(".wav"):
                    x = self.determine_sample_mapping(filename)
                    if x!=None:
                        isbank,idx,notenum,nuance,duration,description = x
                        if len(notenum)>1:
                            notenums=range(notenum[0],notenum[1]+1)
                        else:
                            notenums=notenum
                        for sample_idx in range(len(notenums)):
                            notenum=notenums[sample_idx]
                            if isbank:
                                model = self.banks[bankidx]
                                metadata = model[idx][notenum][nuance]
                                modeldir = os.path.join(self.mididir,
                                                        f"bank-{bankidx+1}",
                                                        f"prog-{idx+1}",
                                                        f"note-{notenum}",
                                                        f"velocity-{nuance}")
                            else:
                                model = self.kits[kitidx]
                                metadata = model[notenum][nuance]
                                modeldir = os.path.join(self.mididir,
                                                        f"kit-{kitidx+1}",
                                                        f"note-{notenum}",
                                                        f"velocity-{nuance}")
                            if metadata["sample_file"]==None:
                                # we don't currently have a sample of this kind
                                self.import_sample(os.path.join(dirname,filename),
                                    metadata,
                                    duration,
                                    nuance,
                                    description,
                                    modeldir,
                                    sample_idx,
                                    len(notenums))
                                self.write_sample_meta(metadata,bankidx=bankidx if isbank else None,
                                                    kitidx=kitidx if not isbank else None,
                                                    progidx=idx,
                                                    notenum=notenum,
                                                    velocity=nuance,
                                                    overwrite=True)
                            else:
                                # we want to keep only the "longest" samples
                                existing_dur = PHILHARMONIA_DURATIONS.index(metadata["duration"]) if metadata["duration"] in PHILHARMONIA_DURATIONS else 10
                                new_dur = PHILHARMONIA_DURATIONS.index(duration) if duration in PHILHARMONIA_DURATIONS else 10
                                if existing_dur < new_dur: # smaller is longer :-]
                                    print(f"{os.path.basename(filename)} has a longer version in the bank -> discarding")
                                    continue
                                elif existing_dur >= new_dur:
                                    if (not overwrite) and existing_dur == new_dur:
                                        print(f"{os.path.basename(filename)} is already represented in the bank (use overwrite option to replace)-> discarding")
                                        continue
                                    
                                    existingsample = os.path.join(modeldir,metadata["sample_file"])
                                    if create_backups:
                                        print(f"{existingsample} -> {existingsample}.backup")
                                        try:
                                            os.rename(existingsample,f"{existingsample}.backup")
                                        except:
                                            pass
                                    else:
                                        try:
                                            os.unlink(existingsample)
                                        except:
                                            pass
                                    self.import_sample(os.path.join(dirname,filename),
                                        metadata,
                                        duration,
                                        nuance,
                                        description,
                                        modeldir,
                                        sample_idx,
                                        len(notenums))
                                    self.write_sample_meta(metadata,bankidx=bankidx if isbank else None,
                                                            kitidx=kitidx if not isbank else None,
                                                            progidx=idx,
                                                            notenum=notenum,
                                                            velocity=nuance,
                                                            overwrite=True)
                    else:
                        print(f"{os.path.basename(filename)} cannot be determined as a General MIDI instrument (try renaming the file) -> discarding")
                        continue
                elif not os.path.exists(f"{os.path.splitext(filepath)[0]}.wav"):
                    print(f"{os.path.basename(filename)} needs to be a wav file -> discarding")
                    continue
    
    def remove_synthed_models(self):
        """ Remove all synthesized data. """

        for bank in self.banks.values():
            for progidx in range(128):
                for notenum in range(128):
                    for velocity in VELOCITY_SHORT.keys():
                        meta=bank[progidx][notenum][velocity]
                        if meta['sample_file'] != None and meta['sample_file'].startswith("SYNTHED"):
                            sample_path = os.path.join(self.mididir,
                                        f'bank-{meta["bankidx"]+1}',
                                        f'prog-{progidx+1}',
                                        f'note-{notenum}',
                                        f'velocity-{velocity}',
                                        meta['sample_file'])
                            print(f"Removing {sample_path}")
                            os.unlink(sample_path)
                            meta['sample_file']=None
                            self.write_sample_meta(meta,
                                                    meta["bankidx"],
                                                    None,
                                                    progidx,
                                                    notenum,
                                                    velocity,
                                                    True)

    def synthesize_missing_samples(self):
        """ Scan banks and create missing model data via pitch shifting.
        
        Currently, only missing pitches are synthesized, while volume (velocity) is synthed on demand;
        see `select_model_data` for more details. Not all 128 pitches will be available for all
        instruments: pitches are only created at a distance at most 12 semitones away from any sample. """

        self.model_data_cache={}
        for bank in self.banks.values():
            for progidx in range(128):
                for notenum in range(128):
                    sample_exists=False
                    for nearest_notenum in np.abs(np.arange(128)-notenum).argsort():
                        for velocity in VELOCITY_SHORT.keys():
                            note_meta=bank[progidx][nearest_notenum][velocity]
                            if note_meta['sample_file'] != None and \
                                    (notenum==nearest_notenum or not note_meta['sample_file'].startswith("SYNTHED")):
                                sample_exists=True
                                break
                        if sample_exists:
                            if nearest_notenum == notenum:
                                break
                            semitones = notenum-nearest_notenum
                            if abs(semitones) <= 12:
                                for velocity in VELOCITY_SHORT.keys():
                                    meta=bank[progidx][nearest_notenum][velocity]
                                    sample_file:str = meta['sample_file']
                                    if sample_file != None and not sample_file.startswith("SYNTHED"):
                                        sample_path = os.path.join(self.mididir,
                                            f'bank-{meta["bankidx"]+1}',
                                            f'prog-{progidx+1}',
                                            f'note-{nearest_notenum}',
                                            f'velocity-{velocity}',
                                            meta['sample_file'])
                                        print(f"Synthesizing bank-{meta['bankidx']} prog-{progidx} note-{notenum} velocity-{velocity} by shifting {semitones} semitones")
                                        model_data = self.read_sample(sample_path,True,True)
                                        pitchshift = PitchShift(semitones,
                                                                use_buffering=False,
                                                                sample_rate=self.sample_rate,
                                                                chunk_size=self.chunk_size,
                                                                dtype=self.dtype,
                                                                polled=True)
                                        pitchshift.receive_signal(model_data)
                                        pitchshift.process_all()
                                        shifted_data = pitchshift.get_out_buf().get_all()
                                        shifted_data /= np.max(np.abs(shifted_data))
                                        shifted_data *= float(VELOCITY_SHORT[velocity])/127.0
                                        new_meta={
                                                    'bankidx':meta['bankidx'],
                                                    'progidx':progidx,
                                                    'note':notenum,
                                                    'velocity':velocity,
                                                    'duration':meta["duration"],
                                                    'sample_file':f'SYNTHED_{meta["sample_file"]}',
                                                    'description':f"Synthesized. {meta['description']}"
                                                }
                                        
                                        self.write_sample_meta(new_meta,
                                                            meta["bankidx"],
                                                            None,
                                                            progidx,
                                                            notenum,
                                                            velocity,
                                                            True)
                                        shifted_path = os.path.join(self.mididir,
                                                                    f'bank-{meta["bankidx"]+1}',
                                                                    f'prog-{progidx+1}',
                                                                    f'note-{notenum}',
                                                                    f'velocity-{velocity}',
                                                                    new_meta['sample_file'])
                                        self.write_model_data(shifted_path,shifted_data)
                                        bank[progidx][notenum][velocity]=new_meta
                            break # check the next notenum
                    if not sample_exists:
                        break # the instrument has no samples 
    
    def select_model_data(self,
            bankidx:int=None,
            progidx:int=None,
            kitidx:int=None,
            note_num:int=None,
            velocity:int=None,
            pressure:int=None,
            fallback_default_bank:bool=True,
            fallback_default_kit:bool=True,
            fallback_nearest_prog:bool=True,
            fallback_default_prog:bool=True):
        """ Return the model data for the requested instrument note.
        
        The nearest velocity sample for the requested instrument note is scaled
        to provide the exact requested velocity. `pressure` is ignored. """
        
        key = (bankidx,
                progidx,
                kitidx,
                note_num,
                velocity,
                pressure,
                fallback_default_bank,
                fallback_default_kit,
                fallback_nearest_prog,
                fallback_default_prog)

        if key in self.model_data_cache:
            return self.model_data_cache[key]

        if bankidx != None:
            if bankidx not in self.banks and fallback_default_bank:
                bankidx = 0
            if bankidx in self.banks:
                bank=self.banks[bankidx]
                programs=[progidx]
                if fallback_nearest_prog:
                    programs+=list(NEAREST_PROGRAMS[GM_PROG_NAMES[progidx]])
                if fallback_default_prog:
                    programs+=[0]
                for progidx in programs:
                    program = bank[progidx]
                    note = program[note_num]
                    for velocity_name in [VELOCITY_NAMES[velocity]] + NEAREST_VELOCITY_SHORT[VELOCITY_NAMES[velocity]]:
                        meta=note[velocity_name]
                        if meta['sample_file'] != None:
                            sample_path = os.path.join(self.mididir,
                                f'bank-{bankidx+1}',
                                f'prog-{progidx+1}',
                                f'note-{note_num}',
                                f'velocity-{velocity_name}',
                                meta['sample_file'])
                            #print("selecting",meta)
                            model_data = self.read_sample(sample_path,False,True)
                            #print(np.max(model_data))
                            velocity_val = VELOCITY_SHORT[meta['velocity']]
                            model_data *= float(velocity)/velocity_val
                            #print("velocity scale",float(velocity)/velocity_val)
                            self.model_data_cache[key]=model_data[:,0]
                            return model_data[:,0]
                return None # no sample for that instrument note
            else:
                return None # no bank with that number
        
        if kitidx != None:
            if kitidx not in self.kits and fallback_default_kit:
                kitidx = 0
            if kitidx in self.kits:
                kits_to_try=[kitidx]
                if kitidx != 0 and kitidx in ROLAND_KITS:
                    if ROLAND_KITS[kitidx]['percussive_names'][note_num]==ROLAND_KITS[0]['percussive_names'][note_num]:
                        kit0_match=True
                        kits_to_try+=[0]
                # Use the sample for the given kit if the sample is available,
                # and use the same named instrument in kit 0 if the sample is not available.
                for kitidx2 in kits_to_try:
                    kit=self.kits[kitidx2]
                    note = kit[note_num]
                    for velocity_name in [VELOCITY_NAMES[velocity]] + NEAREST_VELOCITY_SHORT[VELOCITY_NAMES[velocity]]:
                        meta=note[velocity_name]
                        if meta['sample_file'] != None:
                            sample_path = os.path.join(self.mididir,
                                f'kit-{kitidx2+1}',
                                f'note-{note_num}',
                                f'velocity-{velocity_name}',
                                meta['sample_file'])
                            #print("selecting",meta)
                            model_data = self.read_sample(sample_path,False,True)
                            velocity_val = VELOCITY_SHORT[meta['velocity']]
                            model_data *= float(velocity)/velocity_val
                            self.model_data_cache[key]=model_data[:,0]
                            return model_data[:,0]
                    if kitidx2 != 0 and not kit0_match:
                        break

                return None
            else:
                return None
        
        raise Exception("Either a bank or kit index must be supplied.")

    
    

