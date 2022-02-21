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

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from multiprocessing import Array, Process, Queue, Event, Manager
import ctypes
import os
from multiprocessing.sharedctypes import RawArray
import time
from pathlib import Path


import midiparser
import midiparser.Events
from midiparser.Events.meta import MetaEventKinds
from midiparser.Events.messages import NoteMessage, PressureMessage, PitchBendMessage, \
        ProgramMessage, ChannelPressureMessage
from midiparser.Events.messages.controls import ControlMessage, ControlMessages


from audiomodule.audiomodule import MODULE_ERROR, \
    MODULE_LOG, \
    AM_COMPLETED, \
    AM_CONTINUE, \
    AM_CYCLIC_UNDERRUN, \
    AM_ERROR, \
    AM_INPUT_REQUIRED, \
    AV_RATE, AudioModule, Buffer, audiomod, sw_dtype, ModId

from audiomods.midi_support import GM_PERCUSSIVE_NAMES, NOTE_NAMES, MIDISupport


if TYPE_CHECKING:
    from multiprocessing import _EventType

CONCURRENT_VOICES = 128
CHUNK_BUFFER_LENGTH = 16 # 16 chunks in the shared buffer

class QueueProxy():
    def __init__(self,
            to_queue:Queue):
        self.to_queue=to_queue
        self.proxy=Queue()
    def flush(self):
        while not self.proxy.empty():
            self.to_queue.put(self.proxy.get())
    def proxy(self):
        return self.proxy

def reader_buf(buf:int=0|1,slot:int=0):
    return CONCURRENT_VOICES*buf+slot

def writer_buf(buf:int=0|1,slot:int=0):
    return CONCURRENT_VOICES*(buf+2)+slot

class Synthesizer():
    def __init__(self,model_dir:str,
        sample_rate:int,
        chunk_size:int,
        dtype,
        from_midi:Queue,
        to_midi:Queue,
        synth_data_buffer:np.ndarray,
        buffer_events:list[_EventType]):
        self.model_dir=model_dir
        self.sample_rate=sample_rate
        self.chunk_size=chunk_size
        self.dtype=dtype
        self.from_midi=from_midi
        self.to_midi=to_midi
        self.synth_data_buffer=synth_data_buffer
        self.buffer_events=buffer_events
        
        self.midi_support = MIDISupport(self.model_dir,
                                        sample_rate=self.sample_rate,
                                        chunk_size=self.chunk_size,
                                        dtype=self.dtype)

        self.num_slots = len(self.buffer_events) // 4
        self.current_buffer = np.zeros(self.num_slots,dtype=np.int64)
        self.model_idx = np.zeros(self.num_slots,dtype=np.int64)
        self.output_idx = np.zeros(self.num_slots,dtype=np.int64)
        self.slots_assigned:list[int]=[]
        self.slot_model = [None]*self.num_slots
        self.current_slot:int=None
        self.slot_ready = [False] * self.num_slots
        self.running=False

    def start(self,data:tuple[int,int,int,int,int,int,int,int]):
        slot,bankidx,kitidx,progidx,notenum,velocity,pressure,chunk_idx_start= data
        self.current_buffer[slot]=0
        self.model_idx[slot]=0
        self.output_idx[slot]=chunk_idx_start
        self.slots_assigned.append(slot)
        if self.current_slot == None:
            self.current_slot=0
        model_data=self.midi_support.select_model_data(bankidx,
                                                                progidx,
                                                                kitidx,
                                                                notenum,
                                                                velocity,
                                                                pressure)
        
        self.slot_model[slot] = model_data
        self.slot_ready[slot] = False
        self.current_slot=len(self.slots_assigned)-1

    def stop(self,slot:int):
        if slot in self.slots_assigned:
            buffer = self.current_buffer[slot]
            self.buffer_events[reader_buf(buffer,slot)].set()
            self.slots_assigned.remove(slot)
            if self.current_slot != None and len(self.slots_assigned)>0:
                self.current_slot = self.current_slot % len(self.slots_assigned)
            else:
                self.current_slot= None
            self.to_midi.put(("stopped",slot))

    def reset(self):
        self.slots_assigned=[]
        self.current_slot = None
        self.to_midi.put(("reset",None))

    def quit(self):
        pass

    def run(self):
        self.running=True

    def pause(self):
        self.running=False

    def process_next_slot(self):
        if not self.running:
            time.sleep(0.001)
            return
        if self.current_slot!=None:
            slot = self.slots_assigned[self.current_slot]
            buffer = self.current_buffer[slot]
            if not self.slot_ready[slot]:
                #print(f"waiting for slot {slot} buffer {buffer}")
                if self.buffer_events[writer_buf(buffer,slot)].is_set():
                    self.slot_ready[slot]=True
                    self.buffer_events[writer_buf(buffer,slot)].clear()
            finished=False
            if self.slot_ready[slot]:
                #print(f"processing slot {slot}")
                model_idx = self.model_idx[slot] 
                output_idx = self.output_idx[slot]
                slot_model = self.slot_model[slot]
                #print(slot_model[::1024])
                
                remaining_chunk_size = self.chunk_size*CHUNK_BUFFER_LENGTH - (output_idx % self.chunk_size)
                if type(slot_model)!=type(None) and model_idx<len(slot_model):
                    consumed = min(remaining_chunk_size,len(slot_model)-model_idx)
                    self.synth_data_buffer[buffer,slot,output_idx:output_idx+consumed] = \
                            slot_model[model_idx:model_idx+consumed]
                    self.model_idx[slot]+=consumed
                    if self.model_idx[slot] >= len(slot_model):
                        finished=True
                else:
                    consumed=0
                if consumed < remaining_chunk_size:
                    remaining = remaining_chunk_size-consumed
                    self.synth_data_buffer[buffer,slot,output_idx+consumed:output_idx+consumed+remaining] = 0
                
                

                #print(self.synth_data_buffer[buffer,slot,output_idx:output_idx+self.chunk_size])
                self.output_idx[slot]+=remaining_chunk_size
                
                
                if self.output_idx[slot]==self.chunk_size * CHUNK_BUFFER_LENGTH:
                    #print(self.synth_data_buffer[buffer,slot,::1024])
                    self.buffer_events[reader_buf(buffer,slot)].set()
                    self.current_buffer[slot]=1-buffer
                    self.slot_ready[slot]=False
                    self.output_idx[slot]=0
                
            if finished:
                #self.buffer_events[reader_buf(buffer,slot)].set()
                self.slots_assigned.remove(slot)
                if self.current_slot != None and len(self.slots_assigned)>0:
                    self.current_slot = self.current_slot % len(self.slots_assigned)
                else:
                    self.current_slot= None
                self.to_midi.put(("finished",slot))
            else:
                self.current_slot = (self.current_slot+1) % len(self.slots_assigned)

def synth_process(model_dir:str,
        sample_rate:int,
        chunk_size:int,
        dtype,
        from_midi:Queue,
        to_midi:Queue,
        synth_data_buffer:np.ndarray,
        *argv):
    synth = Synthesizer(model_dir,
        sample_rate,
        chunk_size,
        dtype,
        from_midi,
        to_midi,
        synth_data_buffer,
        argv)
    to_midi.put(("ready",None))
    command,data = from_midi.get()
    while True:
        if command == 'start':
            # start synthesizing
            synth.start(data)
        elif command == 'stop':
            # stop synthesizing
            slot = data
            synth.stop(slot)
        elif command == 'reset':
            # reset all synthesizing
            synth.reset()
        elif command == 'run':
            # synthesizer should start running
            synth.run()
        elif command == 'pause':
            # synthesizer should pause running
            synth.pause()
        elif command == 'quit':
            synth.quit()
            break
        while from_midi.empty():
            synth.process_next_slot()
        command,data = from_midi.get()
    print("synth process finished")

    
def midi16(value:tuple[int,int])->int:
    return value[0]*128+value[1]

def midi16_normalized(value:tuple[int,int])->float:
    return float(midi16(value))/0x3fff

def set_midi16(value:tuple[int,int],LSB=None,MSB=None):
    if LSB and MSB==None:
        return (value[0],LSB)
    elif LSB==None and MSB:
        return (MSB,value[1])
    else:
        return (MSB,LSB)

class MidiKernel(AudioModule):
    """ Read a midi file and output synthesized sounds. """

    name = "MIDI"
    category = "Musical"
    description = ("Basic MIDI support (MIDI 1.0). 16 MIDI channels (module outputs): "
        "3 channels (left, right, center) per module output.")

    def __init__(self,filename:str="",
            model_dir:str="",
            midi_channels:int=16,
            cpus:int=4,
            **kwargs):
        super().__init__(num_inputs=0,
            num_outputs=midi_channels,
            out_chs=[3]*midi_channels,
            **kwargs)
        self.filename=filename
        self.model_dir=model_dir
        self.ready = False
        self.cpus=cpus
        self.chunk_buffer_len = CHUNK_BUFFER_LENGTH # number of chunks to buffer in the synth models
        self.midi_channels=midi_channels
        self.num_instruments=128 # in one bank
        self.num_notes=128 # midi notes or pitches
        self.concurrent_voices=CONCURRENT_VOICES
        self.voice_idx = np.arange(self.concurrent_voices)
        # model idx provides the index into the instrument's synthed samples, for each note on each channel       
        self.last_idx=np.zeros(shape=(self.concurrent_voices),dtype=np.int64)

        # using shared memory for the synth cpus to write to
        self.synth_data = RawArray(np.ctypeslib.as_ctypes_type(self.dtype),
            2*self.concurrent_voices*self.chunk_size*self.chunk_buffer_len)
       
        # The model data provides the samples for every concurrently synthesized instrument instance
        # over all channels and over all note on events per channel.
        # This means that notes must be turned off or else they occupy a slot here for ever and
        # the maximum slots will run out.
        self.synth_instrument_models=np.frombuffer(self.synth_data,dtype=self.dtype,count=2*self.concurrent_voices*self.chunk_size*self.chunk_buffer_len)
        self.synth_instrument_models=np.reshape(self.synth_instrument_models,(2,self.concurrent_voices,
            self.chunk_size*self.chunk_buffer_len))
        self.current_shared_buffer=np.zeros(self.concurrent_voices,dtype=np.int64)
        self.available_slots = set(list(range(self.concurrent_voices)))
        self.required_waits=[]

        # the notes that are currently on (have not been switched off yet) in each channel
        self.notes_on=[set([]) for _ in range(self.midi_channels)]
        # the slot for the currently on notes
        self.note_slots=np.zeros(shape=(self.midi_channels,self.num_notes),dtype=np.int64)
        # the channel and note num for the slot
        self.slot_to_chnote=[None for _ in range(self.concurrent_voices)]
        # the note pressure for the notes that are currently on
        self.note_pressure=np.zeros(shape=(self.midi_channels,self.num_notes),dtype=np.int64)
        self.note_pressure+=127
        # the note velocity for the notes that are currently on
        self.note_velocity=np.zeros(shape=(self.midi_channels,self.num_notes),dtype=np.int64)
        
        self.damper_pedal=["OFF"]*self.midi_channels
        self.channel_pan=[(64,0)]*self.midi_channels
        self.channel_volume=[(100,0)]*self.midi_channels
        self.channel_pressure=[127]*self.midi_channels
        self.channel_program=[0]*self.midi_channels
        self.pitch_bend=[0]*self.midi_channels
        self.channel_balance=[(64,0)]*self.midi_channels
        self.channel_bankkit=[(0,0)]*self.midi_channels
        self.channel_modulation_wheel=[(0,0)]*self.midi_channels
        self.channel_breath_controller=[(0,0)]*self.midi_channels
        self.channel_foot_pedal=[(0,0)]*self.midi_channels
        self.channel_portamento_time=[(0,0)]*self.midi_channels
        self.channel_expression=[(127,127)]*self.midi_channels

        # track data
        self.midi_tracks:dict[int,midiparser.Track]={}
        self.midi_track_idx={}
        self.midi_track_name={}
        self.midi_track_instrument_name={}
        self.midi_track_time={}
        self.midi_ticks_per_quarter:int=48
        self.midi_tempo:float=500000 # us per quarter note
        self.tick_duration:float= self.midi_tempo/self.midi_ticks_per_quarter/1000000.0
        self.clocksPerTick = 24
        self.midi_clock_duration=1.0/(((24.0*500000.0/self.midi_tempo*120.0)/60.0)/1000.0)
        self.current_tick = 0
        self.chunk_duration = self.chunk_size/self.sample_rate
        self.current_chunk = 0 # the current chunk number over all time
        self.current_chunk_idx = 0 # the index into the currently processed chunk
        self.current_event_idx=0
        self.slots_to_reuse = set([]) # the slots that can be reused after the chunk
        self.slots_discarded = set([]) # slots that will not read any more data until they are reclaimed
        
        self.output_buffer=[[] for _ in range(self.midi_channels)]

        self.manager = Manager()
        self.to_synth_q=[]
        self.from_synth_q=[]
        self.pool=[]
        self.buffer_events=[]

    def reset_controllers(self,channel=None):
        """ Reset the controller values to their defaults for all channels, or for the given channel. """

        channels = range(self.midi_channels) if channel==None else [channel]
        for ch in channels:
            self.damper_pedal[ch]="OFF"
            self.channel_pan[ch]=(64,0)
            self.channel_volume[ch]=(100,0)
            self.channel_pressure[ch]=127
            self.channel_program[ch]=0
            self.pitch_bend[ch]=0
            self.channel_balance[ch]=(64,0)
            self.channel_bankkit[ch]=(0,0)
            self.channel_modulation_wheel[ch]=(0,0)
            self.channel_breath_controller[ch]=(0,0)
            self.channel_foot_pedal[ch]=(0,0)
            self.channel_portamento_time[ch]=(0,0)
            self.channel_expression[ch]=(127,127)

    def start(self):
        super().start()
        for q in self.to_synth_q:
            print("sending run to synth process")
            q.put(("run",None))

    
    def stop(self):
        super().stop()
        for q in self.to_synth_q:
            print("sending pause to synth process")
            q.put(("pause",None))

    def open(self):
        if len(self.pool)>0:
            return
        super().open()
        print("opening and starting synth processes")
        self.midi_tracks={}
        self.midi_track_idx={}
        self.midi_track_name={}
        self.midi_track_instrument_name={}
        self.open_midi_file()
        # synth process pool
        self.pool:list[Process] = []
        self.to_synth_q:list[Queue] = []
        self.from_synth_q:list[Queue] = []
        self.assigned_slots:list[set] = []
        self.slot_to_cpu=[-1]*self.concurrent_voices
        self.buffer_events:list[_EventType] = [self.manager.Event() for _ in range(self.concurrent_voices*4)]
        for cpu in range(self.cpus):
            self.assigned_slots.append(set([]))
            to_synth = self.manager.Queue()
            from_synth = self.manager.Queue()
            process = Process(target=synth_process,args=tuple([self.model_dir,
                self.sample_rate,
                self.chunk_size,
                self.dtype,
                to_synth,from_synth,
                self.synth_instrument_models]+self.buffer_events))
            self.pool.append(process)
            self.to_synth_q.append(to_synth)
            self.from_synth_q.append(from_synth)
            process.start()
        for cpu in range(self.cpus):
            print("waiting for ready message from synth process")
            self.from_synth_q[cpu].get() # wait for the ready message
        self.required_waits=[]
        self.slots_discarded = set([])

    def close(self):
        super().close()
        for event in self.buffer_events:
            event.set()
        self.reset()
        for q in self.to_synth_q:
            print("sending quit to synth process")
            q.put(("quit",None))
        #for q in self.from_synth_q:
        #    print("closing from synth queues")
        #    q.close()
        for c in self.pool:
            print("joining with synth process")
            c.join()
            print("closing synth process")
            c.close()
        self.to_synth_q=[]
        self.from_synth_q=[]
        self.pool=[]


    def reset(self):
        """ Reset the MIDI device, equivalent to switching the power off and on (power cycle). """

        print("resetting")
        if len(self.to_synth_q)>0:
            for cpu in range(self.cpus):
                print("sending reset to synth processes")
                self.to_synth_q[cpu].put(("reset",None))
            for cpu in range(self.cpus):
                print("checking synth queues")
                while not self.from_synth_q[cpu].empty():
                    print("waiting for command from synth process")
                    command,data = self.from_synth_q[cpu].get()
                    if command=="stopped":
                        print("received stopped")
                        self.available_slots.add(data)
                        self.assigned_slots[cpu].remove(data)
                        self.slot_to_cpu[data]=-1
                    elif command=="reset":
                        print("received reset")
                        continue
        self.output_buffer=[[] for i in range(self.midi_channels)]
        self.current_chunk = 0
        self.current_tick = 0
        self.notes_on=[set([]) for _ in range(self.midi_channels)]
        self.note_slots=np.zeros(shape=(self.midi_channels,self.num_notes),dtype=np.int64)
       
        self.reset_controllers()

        self.last_idx=np.zeros(shape=(self.concurrent_voices),dtype=np.int64)
        self.current_event_idx=0
        self.required_waits=[]
        self.slots_discarded = set([])
       
        print("finished resetting")

    def set_time_params(self):
        tick_duration_us = self.midi_tempo/self.midi_ticks_per_quarter
        self.tick_duration = tick_duration_us/1000000.0
        self.midi_clock_duration=1.0/(((24.0*500000.0/self.midi_tempo*120.0)/60.0)/1000.0)
        #print(f"\nMidi clock duration {self.midi_clock_duration}")
        #print(f"\nMidi tick duration {self.tick_duration}")

    def start_synthesizing_data(self,
            slot:int,
            program:int,
            note_num:int,
            velocity:int,
            pressure:int,
            channel:int,
            chunk_idx_start:int):
        """ Allocate the synth job to the least loaded synth process. """
        
        min_load=self.concurrent_voices+1
        min_cpu=0
        for cpu in range(self.cpus):
            load = len(self.assigned_slots[cpu])
            if load<min_load:
                min_cpu=cpu
                min_load=load
        # Clear all of the events for the slot, except the
        # writer event for buf 0, which signals the writer
        # that is allowed to write to the buf.
        self.buffer_events[reader_buf(0,slot)].clear()
        self.buffer_events[reader_buf(1,slot)].clear()
        self.buffer_events[writer_buf(1,slot)].clear()
        self.buffer_events[writer_buf(0,slot)].set() # synth process is allowed to write here
        self.slot_to_cpu[slot]=min_cpu
        self.current_shared_buffer[slot]=0 
        self.assigned_slots[min_cpu].add(slot)
        #print("assigned slot to cpu",slot,min_cpu)
        # TODO: bankidx and kitidx should be set per channel
        self.to_synth_q[min_cpu].put(("start",(slot,
                                                0 if channel!=9 else None,
                                                0 if channel==9 else None,
                                                program,
                                                note_num if channel==9 else note_num,
                                                velocity,
                                                pressure,
                                                chunk_idx_start)))
        self.required_waits.append((0,slot))
        

    def stop_synthesizing_data(self,slot:int):
        self.to_synth_q[self.slot_to_cpu[slot]].put(("stop",slot))

    def compute_chunk_data(self,chunk_idx):
        """ Compute the chunk data up to but not including chunk_idx.
        
        `chunk_idx` must be greater than 0. """

        cci = self.current_chunk_idx
        idx = np.arange(chunk_idx-cci)
        last_idx = self.last_idx
        
        # wait for any of the synths that are starting up
        for buf,slot in self.required_waits:
            self.buffer_events[reader_buf(buf,slot)].wait()
            self.buffer_events[reader_buf(0,slot)].clear()
            self.buffer_events[writer_buf(1,slot)].set()
        self.required_waits=[]

        for ch in range(self.midi_channels):
            vol = midi16_normalized(self.channel_volume[ch])
            pres = float(self.channel_pressure[ch])
            pan_right = midi16_normalized(self.channel_pan[ch])
            pan_left = 1.0 - pan_right
            for note_num in self.notes_on[ch]:
                slot = self.note_slots[ch,note_num]
                model_data = self.synth_instrument_models[self.current_shared_buffer[slot],slot,:]
                fill = idx+last_idx[slot]
                self.chunk_output[ch,cci:chunk_idx,0]=model_data[fill]*vol*pan_left*self.note_pressure[ch,note_num]/pres
                self.chunk_output[ch,cci:chunk_idx,1]=model_data[fill]*vol*pan_right*self.note_pressure[ch,note_num]/pres
                self.chunk_output[ch,cci:chunk_idx,2]=(self.chunk_output[ch,cci:chunk_idx,0]+self.chunk_output[ch,cci:chunk_idx,1])*0.5
                self.last_idx[slot]=fill[-1]+1
        self.current_chunk_idx = chunk_idx

    def set_note_on(self,channel,note_number,velocity,chunk_idx_start):
        """ Set the note on. """

        notes_on = self.notes_on[channel]
        if velocity==0:
            # the note should stop sustaining, unless the damper_pedal is on (hold), etc.
            return
        elif note_number in notes_on:
            # the note is being struck again, so the existing note is no longer rendered
            self.mute_note(channel,note_number)
        # the note is struck with the given velocity
        self.notes_on[channel].add(note_number)
        if len(self.available_slots)==0:
            raise Exception("Polyphonic voices exhausted")
        # if a slot is in the available slots set, it is guaranteed to
        # be unused by all synth processes
        slot = self.available_slots.pop()
        self.note_slots[channel,note_number]=slot
        self.last_idx[slot]=chunk_idx_start # reset to the start of the synthed model data
        self.slot_to_chnote[slot]=(channel,note_number)
        self.start_synthesizing_data(slot,
            self.channel_program[channel],
            note_number,
            velocity,
            self.note_pressure[channel,note_number],
            channel,
            chunk_idx_start)
        
    def set_note_off(self,channel,note_number):
        """ The note is allowed to release as per its ADSR envelope.
        
        Thus the note is still rendered in the channel, and the synthesizer
        process determines when the signal has completely faded."""

        pass
    
    def mute_note(self,channel,note_number):
        """ The note is turned off and immediately muted. 
        
        The note will no longer render in the channel, and the synthesizer
        process is told to stop synthesizing it."""

        notes_on =self.notes_on[channel]
        if note_number in notes_on:
            # this note is no longer rendering on this channel
            self.notes_on[channel].remove(note_number)
            slot = self.note_slots[channel,note_number]
            # we need to set last idx to 0 so we don't try to swap buffers on a slot
            # that is being discarded
            self.last_idx[slot]=0
            # discarding a slot means that we are not interested in any further data
            # produced on this slot, until we reuse the slot for some other synth
            self.slots_discarded.add(slot)
            # we can't add the slot back to the pool until the synth process has stopped
            # using it
            self.stop_synthesizing_data(slot)
        else:
            self.observer.put((MODULE_LOG,(self.mod_id,"error",f"MIDI note {note_number} on channel {channel} being set off was not on."),None))

    def set_note_pressure(self,channel,note_number,pressure):
        self.note_pressure[channel,note_number]=pressure

    def set_pressure(self,midi_channel,pressure):
        """ Set the midi channel pressure. """
        self.channel_pressure[midi_channel]=pressure

    def set_program(self,midi_channel,program_num):
        """ Set the midi channel to use the program (instrument). """
        self.channel_program[midi_channel]=program_num

    def set_pitch_bend(self,midi_channel,value):
        """ Set the pitch bend for the channel. """
        self.pitch_bend[midi_channel]=value

    #
    # Interpretation of control messages adapted from
    # http://midi.teragonaudio.com/tech/midispec.htm
    #

    def set_bank_select(self,channel,LSB=None,MSB=None):
        """ Select the bank/kit for the channel."""
        self.channel_bankkit[channel]=set_midi16(self.channel_bankkit[channel],LSB,MSB)

    def set_modulation_wheel(self,channel,LSB=None,MSB=None):
        """ Set the modulation wheel value.
        
        Usually a vibrato effect. 0 is no modulation."""
        self.channel_modulation_wheel[channel]=set_midi16(self.channel_modulation_wheel[channel],LSB,MSB)

    def set_breath_controller(self,channel,LSB=None,MSB=None):
        """ Set the breath conroller value. 
        
        0 is minimum breath pressure."""
        self.channel_breath_controller[channel]=set_midi16(self.channel_breath_controller[channel],LSB,MSB)

    def set_foot_pedal(self,channel,LSB=None,MSB=None):
        """ Set the foot controller value.
        
        0 is minimum effect."""
        self.channel_foot_pedal[channel]=set_midi16(self.channel_foot_pedal[channel],LSB,MSB)

    def set_portamento_time(self,channel,LSB=None,MSB=None):
        """ The rate at which portamento slides the pitch between 2 notes. 
        
        0 is the slowest rate."""
        self.channel_portamento_time[channel]=set_midi16(self.channel_portamento_time[channel],LSB,MSB)

    def set_data_entry(self,channel,LSB=None,MSB=None):
        """ Set the value of the parameter selected by a previous RPN or NRPN. """
        pass

    def set_volume(self,midi_channel,LSB=None,MSB=None):
        """ Set the midi channel volume.
        
        Master volume may be controlled by another method
        such as the Universal SysEx Master Volume message, or
        take its volume from one of the Parts, or be controlled by
        a General Purpose Slider controller.
        
        Expression Controller also may affect the volume.
        
        The actual volume should be 40log(volume/127) (coarse only)
        or 40log(volume/(127**2)) (fine), where volume may be 
        multiplied by expression."""
        self.channel_volume[midi_channel] = set_midi16(self.channel_volume[midi_channel],LSB,MSB)
        

    def set_balance(self,channel,LSB=None,MSB=None):
        """ Set the device's stereo balance. 
        
        Typically balance is used to adjust the volume of stereo
        elements without changing their pan positions.
        
        0x2000/64 is center for fine/coarse. 0 is left emphasis."""
        self.channel_balance[channel] = set_midi16(self.channel_balance[channel],LSB,MSB)

    def set_pan(self,channel,LSB=None,MSB=None):
        """ Set where within the stereo field the device's sound
        will be placed.
        
        Pan effects all notes on the channel, including notes
        that were triggered prior to the pan message being received,
        and are still sustaining.
        
        0x2000/64 is center for fine/coarse. 0 is hard left. """
        self.channel_pan[channel] = set_midi16(self.channel_pan[channel],LSB,MSB)

    def set_expression(self,channel,LSB=None,MSB=None):
        """ Set the percentage of Volume (as set by the Volume 
        Controller).
        
        Expression is used for doing crescendos and decrescendos.
        
        With coarse value, 63 is 50%."""
        self.channel_expression[channel]=set_midi16(self.channel_expression[channel],LSB,MSB)

    def set_effect_controller_1(self,channel,LSB=None,MSB=None):
        """ Set a parameter relating to an effects device. """
        pass

    def set_effect_controller_2(self,channel,LSB=None,MSB=None):
        """ Set a parameter relating to an affects device. """
        pass

    def set_general_purpose_1(self,channel,value):
        """ Set general purpose slider 1 value.
        
        0 is minimum effect."""
        pass

    def set_general_purpose_2(self,channel,value):
        """ Set general purpose slider 2 value.
        
        0 is minimum effect."""
        pass

    def set_general_purpose_3(self,channel,value):
        """ Set general purpose slider 3 value.
        
        0 is minimum effect."""
        pass

    def set_general_purpose_4(self,channel,value):
        """ Set general purpose slider 4 value.
        
        0 is minimum effect."""
        pass

    def set_damper_pedal(self,channel,value):
        """ Set the damper pedal on the channel.
        
        When on, this holds (i.e., sustains) notes that are playing
        or played,
        even if the musician releases the notes. The Note Off effect
        is postponed until the musician switches the damper pedal
        off."""

        self.damper_pedal[channel]=value
        if value=="OFF":
            notes_on = list(self.notes_on[channel])
            for note in notes_on:
                self.set_note_off(channel,note)

    def set_portamento_pedal(self,channel,value):
        """ Set the portamento effect on the channel. """
        pass

    def set_sostenuto_pedal(self,channel,value):
        """ Similar to the damper pedal (hold), except this only
        sustains notes that are already on when the pedal is
        turned on.
        
        After the pedal is on, it continues to hold these initial
        notes all of the while that the pedal is on, but during
        that time, all other arriving Note Ons are not held."""
        pass

    def set_soft_pedal(self,channel,value):
        """ Set the soft pedal on the channel.
        
        Lowers the volume of any notes played."""
        pass

    def set_legato_pedal(self,channel,value):
        """ Set the legato effect between notes, which is usually
        achieved by skipping the attack portion of the VCA's envelope."""
        pass

    def set_hold_2(self,channel,value):
        """ Set the hold 2 pedal on the channel.
        
        This lengthens the release time of the playing notes' VCA. 
        Unlike the hold (damper pedal) controller, this pedal doesn't
        permantently sustain the note's sound. """
        pass

    def set_sound_variation(self,channel,value):
        """ Affects any parameter associate with the circuitry that
        produces sound.
        
        This controller may adjust the sample rate, i.e. playback
        speed, for a 'tuning' control.
        
        0 is minimum effect."""
        pass

    def set_sound_timbre(self,channel,value):
        """ Controls the filter's envelope levels.
        
        This controls how the filter shapes the 'brightness' of the 
        sound over time.
        
        0 is minimum effect."""
        pass

    def set_sound_release_time(self,channel,value):
        """ Controls the amp's envelope release time, for a control
        over how long it takes a sound to fade out. 
        
        0 is the minimum setting."""
        pass

    def set_sound_attack_time(self,channel,value):
        """ Controls the amp's envelope attack time, for a control
        over how long it takes a sound to fade in.
        
        0 is the minimum setting."""
        pass

    def set_sound_brightness(self,channel,value):
        """ Controls the filter's cutoff frequency, for an overall
        'brightness' control. 
        
        0 is the minimum setting."""
        pass

    def set_sound_control_6(self,channel,value):
        """ Control any parameters associated with the circuitry
        that produces sound. 
        
        0 is the minimum setting."""
        pass

    def set_sound_control_7(self,channel,value):
        """ Control any parameters associated with the circuitry
        that produces sound. 
        
        0 is the minimum setting."""
        pass

    def set_sound_control_8(self,channel,value):
        """ Control any parameters associated with the circuitry
        that produces sound. 
        
        0 is the minimum setting."""
        pass

    def set_sound_control_9(self,channel,value):
        """ Control any parameters associated with the circuitry
        that produces sound. 
        
        0 is the minimum setting."""
        pass

    def set_sound_control_10(self,channel,value):
        """ Control any parameters associated with the circuitry
        that produces sound. 
        
        0 is the minimum setting."""
        pass

    def set_general_purpose_button_1(self,channel,value):
        """ On/off buttons for control of an affect. """
        pass

    def set_general_purpose_button_2(self,channel,value):
        """ On/off buttons for control of an affect. """
        pass

    def set_general_purpose_button_3(self,channel,value):
        """ On/off buttons for control of an affect. """
        pass

    def set_general_purpose_button_4(self,channel,value):
        """ On/off buttons for control of an affect. """
        pass

    def set_effects_level(self,channel,value):
        """ Set the effects level for the channel.
        
        Often this is the reverb or delay level.
        
        0 is no effect applied at all."""
        pass

    def set_tremulo_level(self,channel,value):
        """ Set the tremulo amount for this channel. 
        
        0 is no tremulo."""
        pass

    def set_chorus_level(self,channel,value):
        """ Set the chorus amount for this channel.
        
        0 is no chorus."""
        pass

    def set_celeste_level(self,channel,value):
        """ Set the celeste (detune) level for this channel.
        
        0 is no celeste."""
        pass

    def set_phaser_level(self,channel,value):
        """ Set the phaser level for this channel.
        
        0 is no phaser."""
        pass

    def set_data_button_increment(self,channel,value):
        """ Increase a data button's value, as selected by
        a preceding RPN or NRPN."""
        pass

    def set_data_button_decrement(self,channel,value):
        """ Decrease a data button's value, as selected by
        a preceding RPN or NRPN."""
        pass

    def set_nrpn(self,channel,LSB=None,MSB=None):
        """ Set a NRPN number for use by subsequence Data Button
        Increment/Decrement and Data Entry controllers. """
        pass

    def set_rpn(self,channel,LSB=None,MSB=None):
        """ Set a RPN number for use by subsequence Data Button
        Increment/Decrement and Data Entry controllers. """
        pass

    def set_all_sound_off(self,channel,value):
        """ Mutes all sounding notes that were turned on by received
        Note On messages, and which haven't yet been turned off
        by respective Note Off messages.
        
        Does not mute notes played locally. Immediately mutes notes,
        regardless of VCA fade or damper pedal settings."""
        for channel in range(self.midi_channels):
            notes_on = self.notes_on[channel]
            for note in list(notes_on):
                self.mute_note(channel,note)

    def set_reset_all_controllers(self,channel,value):
        """ Reset all controller values to default values. """
        self.reset_controllers(channel=channel)

    def set_local_control(self,channel,value):
        """ Set the local keyboard to be used or not used (ignored)."""
        if value<=63:
            """Off"""
        else:
            """On"""

    def set_all_notes_off(self,channel,value):
        """ Turns off all notes that were turned on by received
        Note On messages, and which haven't yet been turned off by
        respective Note Off messages.
        
        Does not affect the local keyboard. Does not override a
        hold (damper) pedal."""
        pass

    def set_omni_mode_off(self,channel,value):
        """ Turn omni mode off. """
        pass

    def set_omni_mode_on(self,channel,value):
        """ Turn omni mode on. """
        pass

    def set_mono_mode_on(self,channel,value):
        """ Enables monophonic operation (disables polyphonic operation).
        
        When receiving a this message, all playing notes are turned off."""
        pass

    def set_poly_mode_on(self,channel,value):
        """ Enables polyphonic operation. 
        
        When receiving a this message, all playing notes are turned off."""
        pass

    async def next_chunk(self):
        ret = self.process_next_chunk()
        if ret == AM_ERROR:
            return AM_ERROR
        if len(self.output_buffer[0])>0:
            for out_idx in range(self.num_outputs):
                self.send_signal(self.output_buffer[out_idx].pop(0),out_idx)
            return AM_CONTINUE
        return AM_COMPLETED

    def process_next_chunk(self):
        if not self.ready:
            return AM_ERROR

        # process messages from synth processes - reclaim slots no longer used
        for cpu in range(self.cpus):
            while not self.from_synth_q[cpu].empty():
                command,data = self.from_synth_q[cpu].get()
                if command=="stopped":
                    #print("reclaiming slot",data)
                    self.available_slots.add(data)
                    self.assigned_slots[cpu].remove(data)
                    self.slot_to_cpu[data]=-1
                    self.slots_discarded.remove(data)
                    self.slot_to_chnote[data]=None
                elif command=="finished":
                    self.available_slots.add(data)
                    self.assigned_slots[cpu].remove(data)
                    self.slot_to_cpu[data]=-1
                    if data in self.slots_discarded:
                        self.slots_discarded.remove(data)
                    channel,note_number = self.slot_to_chnote[data]
                    if note_number in self.notes_on[channel]:
                        self.notes_on[channel].remove(note_number)
                    self.slot_to_chnote[data]=None
                    self.last_idx[data]=0
                elif command=="midi_error":
                    self.observer.put((MODULE_LOG,(self.mod_id,"error",data),None))

        # find slots that require to swap their shared buffers
        for slot in self.voice_idx[self.last_idx==self.chunk_buffer_len*self.chunk_size]:
            current_buf = self.current_shared_buffer[slot]
            next_buf = 1 - current_buf
            self.buffer_events[reader_buf(next_buf,slot)].wait()
            self.buffer_events[reader_buf(next_buf,slot)].clear()
            self.buffer_events[writer_buf(current_buf,slot)].set()
            self.current_shared_buffer[slot]=next_buf
            self.last_idx[slot]=0
        

        self.current_chunk_idx = 0
        self.chunk_output = np.zeros(shape=(self.midi_channels,self.chunk_size,3))

        if self.current_event_idx==len(self.all_midi_events):
            return AM_COMPLETED

        # process midi events for this chunk
        start_of_next_chunk = (self.current_chunk+1)*self.chunk_duration
        #print(start_of_next_chunk)
        tracks_to_remove = set([])
       
        while self.current_event_idx<len(self.all_midi_events) and \
                self.all_midi_events[self.current_event_idx][0]*self.tick_duration < start_of_next_chunk:
            time,track_id,track_idx = self.all_midi_events[self.current_event_idx]
            self.current_event_idx+=1
            time_s = time*self.tick_duration
            self.current_tick = time
            # the chunk_idx is the index into the current chunk
            # where the current midi event is located
            chunk_idx = int(time_s/self.sample_duration) % self.chunk_size
            # compute the chunk data up to (but not including) the current sample
            if chunk_idx > self.current_chunk_idx:
                self.compute_chunk_data(chunk_idx)
            event = self.midi_tracks[track_id][track_idx]
            track_name=self.midi_track_name[track_id]
            if isinstance(event,midiparser.Events.MetaEvent):
                log_prefix=f"MIDI track {track_id} - {track_name}:"
                if event.message == MetaEventKinds.Copyright_Notice:
                    copyright_notice = event.attributes["text"]
                    log_suffix=f"Copyright Notice {copyright_notice}"
                elif event.message == MetaEventKinds.Cue_Point:
                    cue_point = event.attributes["text"]
                    log_suffix=f"Cue Point {cue_point}"
                elif event.message == MetaEventKinds.End_Of_Track:
                    log_suffix="End of Track"
                    tracks_to_remove.add(track_id)
                elif event.message == MetaEventKinds.Instrument_Name:
                    instrument_name = event.attributes["text"]
                    log_suffix=f"Instrument Name {instrument_name}"
                    self.midi_track_instrument_name[track_id] = instrument_name
                elif event.message == MetaEventKinds.Key_Signature:
                    key = event.attributes["key"]
                    mode = event.attributes["mode"]
                    log_suffix=f"Key Signature {key} {mode}"
                elif event.message == MetaEventKinds.Lyric:
                    lyric = event.attributes["text"]
                    log_suffix=f"Lyric {lyric}"
                elif event.message == MetaEventKinds.Marker:
                    marker = event.attributes["text"]
                    log_suffix=f"Marker {marker}"
                elif event.message == MetaEventKinds.MIDI_Channel_Prefix:
                    channel = event.attributes["channel"]
                    log_suffix=f"Channel Prefix {channel}"
                elif event.message == MetaEventKinds.Sequence_Number:
                    sequence_number = event.attributes["number"]
                    log_suffix=f"Sequence Number {sequence_number}"
                elif event.message == MetaEventKinds.Sequencer_Specific:
                    log_suffix=f"Sequencer Specific"
                elif event.message == MetaEventKinds.Set_Tempo:
                    self.midi_tempo = event.attributes["tempo"]
                    self.set_time_params()
                    log_suffix=f"Set Tempo {self.midi_tempo}"
                elif event.message == MetaEventKinds.SMTPE_Offset:
                    hh = event.attributes["hh"]
                    mm = event.attributes["mm"]
                    ss = event.attributes["ss"]
                    frame = event.attributes["frame"]
                    log_suffix=f"SMTPE Offset {hh}:{mm}:{ss}.{frame}"
                elif event.message == MetaEventKinds.Time_Signature:
                    numerator = event.attributes["numerator"]
                    denominator = event.attributes["denominator"]
                    clocksPerTick = event.attributes["clocksPerTick"]
                    demisemiquaverPer24Clocks = event.attributes["demisemiquaverPer24Clocks"]
                    self.clocksPerTick = clocksPerTick
                    log_suffix=f"Time Signature {numerator}/{denominator}"
                elif event.message == MetaEventKinds.Track_Name:
                    name = event.attributes["text"]
                    log_suffix=f"MIDI track {track_id} named {name}"
                    self.midi_track_name[track_id] = name
                elif event.message == MetaEventKinds.Text:
                    text = event.attributes["text"]
                    log_suffix=f"{text}"
                else:
                    print("",end="\r")
                    #print(f"Unknown Meta Event",event)
                self.observer.put((MODULE_LOG,(self.mod_id,"info",
                        f"{log_prefix} {log_suffix}"),None))
            elif isinstance(event,midiparser.Events.SysExEvent):
                print(f"MIDI track {track_id} - {track_name}: Sys Ex")
            elif isinstance(event,midiparser.Events.MIDIEvent):
                command = event.command
                channel = event.channel
                if command == 0x80:
                    # note off
                    note_message:NoteMessage = event.message
                    note = note_message.note
                    note_number = note.note_num
                    self.set_note_off(channel,note_number)
                    if channel!=9:
                        print(f"MIDI track {track_id} - {track_name}: Channel {channel} Note Off {NOTE_NAMES[note_number]}")
                        self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                            f"MIDI track {track_id} - {track_name}: Channel {channel} Note Off {NOTE_NAMES[note_number]}"),None))
                    else:
                        print(f"MIDI track {track_id} - {track_name}: Channel {channel} Percussion Off {GM_PERCUSSIVE_NAMES[note_number-27]}")
                        self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                            f"MIDI track {track_id} - {track_name}: Channel {channel} Percussion Off {GM_PERCUSSIVE_NAMES[note_number-27]}"),None))
                elif command == 0x90:
                    # note on
                    note_message:NoteMessage = event.message
                    note = note_message.note
                    velocity = note_message.velocity
                    note_number = note.note_num
                    #print(f'current time {time_s}')
                    self.set_note_on(channel,note_number,velocity,chunk_idx)
                    if channel!=9:
                        print(f"MIDI track {track_id} - {track_name}: Channel {channel} Note On {NOTE_NAMES[note_number]} Velocity {velocity}")
                        self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                            f"MIDI track {track_id} - {track_name}: Channel {channel} Note On {NOTE_NAMES[note_number]} Velocity {velocity}"),None))
                    else:
                        print(f"MIDI track {track_id} - {track_name}: Channel {channel} Percussion On {GM_PERCUSSIVE_NAMES[note_number-27]} Velocity {velocity}")
                        self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                            f"MIDI track {track_id} - {track_name}: Channel {channel} Percussion On {GM_PERCUSSIVE_NAMES[note_number-27]} Velocity {velocity}"),None))
                elif command == 0xa0:
                    # key pressure
                    pressure_message:PressureMessage = event.message
                    note = pressure_message.note
                    pressure = pressure_message.pressure
                    note_number = note.note_num
                    self.set_note_pressure(channel,note_number,pressure)
                    print(f"MIDI track {track_id} - {track_name}: Channel {channel} Note {NOTE_NAMES[note_number]} Pressure {pressure}")
                    self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                        f"MIDI track {track_id} - {track_name}: Channel {channel} Note {NOTE_NAMES[note_number]} Pressure {pressure}"),None))
                elif command == 0xb0:
                    # control change
                    control_message:ControlMessage = event.message
                    control_command = control_message.command
                    value = control_message.value
                    log_prefix=f"MIDI track {track_id} - {track_name}: Channel {channel}"
                    if control_command == ControlMessages.Bank_Select_MSB:
                        log_suffix=f"Bank Select MSB {value}"
                        self.set_bank_select(channel,MSB=value)
                    elif control_command == ControlMessages.Modulation_Wheel_MSB:
                        log_suffix=f"Modulation Wheel MSB {value}"
                        self.set_modulation_wheel(channel,MSB=value)
                    elif control_command == ControlMessages.Breath_Controller_MSB:
                        log_suffix=f"Breath Controller MSB {value}"
                        self.set_breath_controller(channel,MSB=value)
                    elif control_command == ControlMessages.Foot_Pedal_MSB:
                        log_suffix=f"Foot Pedal MSB {value}"
                        self.set_foot_pedal(channel,MSB=value)
                    elif control_command == ControlMessages.Portamento_Time_MSB:
                        log_suffix=f"Portamento Time MSB {value}"
                        self.set_portamento_time(channel,MSB=value)
                    elif control_command == ControlMessages.Data_Entry_MSB:
                        log_suffix=f"Data Entry MSB {value}"
                        self.set_data_entry(channel,MSB=value)
                    elif control_command == ControlMessages.Channel_Volume_MSB:
                        self.set_volume(channel,MSB=value)
                        log_suffix=f"Volume MSB {value}"
                    elif control_command == ControlMessages.Balance_MSB:
                        self.set_balance(channel,MSB=value)
                        log_suffix=f"Balance MSB {value}"
                    elif control_command == ControlMessages.Pan_MSB:
                        self.set_pan(channel,MSB=value)
                        log_suffix=f"Pan MSB {value}"
                    elif control_command == ControlMessages.Expression_MSB:
                        log_suffix=f"Expression MSB {value}"
                        self.set_expression(channel,MSB=value)
                    elif control_command == ControlMessages.Effect_Controller_1_MSB:
                        log_suffix=f"Effect Controller 1 MSB {value}"
                        self.set_effect_controller_1(channel,MSB=value)
                    elif control_command == ControlMessages.Effect_Controller_2_MSB:
                        log_suffix=f"Effect Controller 2 MSB {value}"
                        self.set_effect_controller_2(channel,MSB=value)
                    elif control_command == ControlMessages.General_Purpose_1:
                        log_suffix=f"General Purpose 1 {value}"
                        self.set_general_purpose_1(channel,value)
                    elif control_command == ControlMessages.General_Purpose_2:
                        log_suffix=f"General Purpose 2 {value}"
                        self.set_general_purpose_2(channel,value)
                    elif control_command == ControlMessages.General_Purpose_3:
                        log_suffix=f"General Purpose 3 {value}"
                        self.set_general_purpose_3(channel,value)
                    elif control_command == ControlMessages.General_Purpose_4:
                        log_suffix=f"General Purpose 4 {value}"
                        self.set_general_purpose_4(channel,value)
                    elif control_command == ControlMessages.Bank_Select_LSB:
                        log_suffix=f"Bank Select LSB {value}"
                        self.set_bank_select(channel,LSB=value)
                    elif control_command == ControlMessages.Modulation_Wheel_LSB:
                        log_suffix=f"Modulation Wheel LSB {value}"
                        self.set_modulation_wheel(channel,LSB=value)
                    elif control_command == ControlMessages.Breath_Controller_LSB:
                        log_suffix=f"Breath Controller LSB {value}"
                        self.set_breath_controller(channel,LSB=value)
                    elif control_command == ControlMessages.Foot_Pedal_LSB:
                        log_suffix=f"Foot Pedal LSB {value}"
                        self.set_foot_pedal(channel,LSB=value)
                    elif control_command == ControlMessages.Portamento_Time_LSB:
                        log_suffix=f"Portamento Time LSB {value}"
                        self.set_portamento_time(channel,LSB=value)
                    elif control_command == ControlMessages.Data_Entry_LSB:
                        log_suffix=f"Data Entry LSB {value}"
                        self.set_data_entry(channel,LSB=value)
                    elif control_command == ControlMessages.Channel_Volume_LSB:
                        log_suffix=f"Channel_Volume LSB {value}"
                        self.set_volume(channel,LSB=value)
                    elif control_command == ControlMessages.Balance_LSB:
                        log_suffix=f"Balance LSB {value}"
                        self.set_balance(channel,LSB=value)
                    elif control_command == ControlMessages.Pan_LSB:
                        log_suffix=f"Pan LSB {value}"
                        self.set_pan(channel,LSB=value)
                    elif control_command == ControlMessages.Expression_LSB:
                        log_suffix=f"Expression LSB {value}"
                        self.set_expression(channel,LSB=value)
                    elif control_command == ControlMessages.Effect_Controller_1_LSB:
                        log_suffix=f"Effect Controller 1 LSB {value}"
                        self.set_effect_controller_1(channel,LSB=value)
                    elif control_command == ControlMessages.Effect_Controller_2_LSB:
                        log_suffix=f"Effect Controller 2 LSB {value}"
                        self.set_effect_controller_2(channel,LSB=value)
                    elif control_command == ControlMessages.Damper_Pedal:
                        log_suffix=f"Damper Pedal {value}"
                        self.set_damper_pedal(channel,value)
                    elif control_command == ControlMessages.Portamento_Pedal:
                        log_suffix=f"Portamento Pedal {value}"
                        self.set_portamento_pedal(channel,value)
                    elif control_command == ControlMessages.Sostenuto_Pedal:
                        log_suffix=f"Sostenuto Pedal {value}"
                        self.set_sostenuto_pedal(channel,value)
                    elif control_command == ControlMessages.Soft_Pedal:
                        log_suffix=f"Soft Pedal {value}"
                        self.set_soft_pedal(channel,value)
                    elif control_command == ControlMessages.Legato_Pedal:
                        log_suffix=f"Legato Pedal {value}"
                        self.set_legato_pedal(channel,value)
                    elif control_command == ControlMessages.Hold_2:
                        log_suffix=f"Hold 2 {value}"
                        self.set_hold_2(channel,value)
                    elif control_command == ControlMessages.Sound_Variation:
                        log_suffix=f"Sound Variation {value}"
                        self.set_sound_variation(channel,value)
                    elif control_command == ControlMessages.Sound_Timbre:
                        log_suffix=f"Sound Timbre {value}"
                        self.set_sound_timbre(channel,value)
                    elif control_command == ControlMessages.Sound_Release_Time:
                        log_suffix=f"Sound Release Time {value}"
                        self.set_sound_release_time(channel,value)
                    elif control_command == ControlMessages.Sound_Attack_Time:
                        log_suffix=f"Sound Attack Time {value}"
                        self.set_sound_attack_time(channel,value)
                    elif control_command == ControlMessages.Sound_Brightness:
                        log_suffix=f"Sound Brightness {value}"
                        self.set_sound_brightness(channel,value)
                    elif control_command == ControlMessages.Sound_Control_6:
                        log_suffix=f"Sound Control 6 {value}"
                        self.set_sound_control_6(channel,value)
                    elif control_command == ControlMessages.Sound_Control_7:
                        log_suffix=f"Sound Control 7 {value}"
                        self.set_sound_control_7(channel,value)
                    elif control_command == ControlMessages.Sound_Control_8:
                        log_suffix=f"Sound Control 8 {value}"
                        self.set_sound_control_8(channel,value)
                    elif control_command == ControlMessages.Sound_Control_9:
                        log_suffix=f"Sound Control 9 {value}"
                        self.set_sound_control_9(channel,value)
                    elif control_command == ControlMessages.Sound_Control_10:
                        log_suffix=f"Sound Control 10 {value}"
                        self.set_sound_control_10(channel,value)
                    elif control_command == ControlMessages.General_Purpose_Button_1:
                        log_suffix=f"General Purpose Button 1 {value}"
                        self.set_general_purpose_button_1(channel,value)
                    elif control_command == ControlMessages.General_Purpose_Button_2:
                        log_suffix=f"General Purpose Button 2 {value}"
                        self.set_general_purpose_button_2(channel,value)
                    elif control_command == ControlMessages.General_Purpose_Button_3:
                        log_suffix=f"General Purpose Button 3 {value}"
                        self.set_general_purpose_button_3(channel,value)
                    elif control_command == ControlMessages.General_Purpose_Button_4:
                        log_suffix=f"General Purpose Button 4 {value}"
                        self.set_general_purpose_button_4(channel,value)
                    elif control_command == ControlMessages.Effects_Level:
                        log_suffix=f"Effects Level {value}"
                        self.set_effects_level(channel,value)
                    elif control_command == ControlMessages.Tremulo_Level:
                        log_suffix=f"Tremulo Level {value}"
                        self.set_tremulo_level(channel,value)
                    elif control_command == ControlMessages.Chorus_Level:
                        log_suffix=f"Chorus Level {value}"
                        self.set_chorus_level(channel,value)
                    elif control_command == ControlMessages.Celeste_Level:
                        log_suffix=f"Celeste Level {value}"
                        self.set_celeste_level(channel,value)
                    elif control_command == ControlMessages.Phaser_Level:
                        log_suffix=f"Phaser Level {value}"
                        self.set_phaser_level(channel,value)
                    elif control_command == ControlMessages.Data_Button_increment:
                        log_suffix=f"Data Button increment {value}"
                        self.set_data_button_increment(channel,value)
                    elif control_command == ControlMessages.Data_Button_decrement:
                        log_suffix=f"Data Button decrement {value}"
                        self.set_data_button_decrement(channel,value)
                    elif control_command == ControlMessages.NRPN_LSB:
                        log_suffix=f"NRPN LSB {value}"
                        self.set_nrpn(channel,LSB=value)
                    elif control_command == ControlMessages.NRPN_MSB:
                        log_suffix=f"NRPN MSB {value}"
                        self.set_nrpn(channel,MSB=value)
                    elif control_command == ControlMessages.RPN_LSB:
                        log_suffix=f"RPN LSB {value}"
                        self.set_rpn(channel,LSB=value)
                    elif control_command == ControlMessages.RPN_MSB:
                        log_suffix=f"RPN MSB {value}"
                        self.set_rpn(channel,MSB=value)
                    elif control_command == ControlMessages.All_Sound_Off:
                        log_suffix=f"All Sound Off {value}"
                        self.set_all_sound_off(channel,value)
                    elif control_command == ControlMessages.Reset_All_Controllers:
                        log_suffix=f"Reset All Controllers {value}"
                        self.set_reset_all_controllers(channel,value)
                    elif control_command == ControlMessages.Local_Control:
                        log_suffix=f"Local Control {value}"
                        self.set_local_control(channel,value)
                    elif control_command == ControlMessages.All_Notes_Off:
                        log_suffix=f"All Notes Off {value}"
                        self.set_all_notes_off(channel,value)
                    elif control_command == ControlMessages.Omni_Mode_Off:
                        log_suffix=f"Omni Mode Off {value}"
                        self.set_omni_mode_off(channel,value)
                    elif control_command == ControlMessages.Omni_Mode_On:
                        log_suffix=f"Omni Mode On {value}"
                        self.set_omni_mode_on(channel,value)
                    elif control_command == ControlMessages.Mono_Mode_On:
                        log_suffix=f"Mono Mode On {value}"
                        self.set_mono_mode_on(channel,value)
                    elif control_command == ControlMessages.Poly_Mode_On:
                        log_suffix=f"Mono Mode Off {value}"
                        self.set_poly_mode_on(channel,value)
                    else:
                        log_suffix=f"Undefined Control Command {control_command} {value}"
                    print(f"{log_prefix} {log_suffix}")
                    self.observer.put((MODULE_LOG,(self.mod_id,"debug",f"{log_prefix} {log_suffix}"),None))
                elif command == 0xc0:
                    # program change
                    program_message:ProgramMessage = event.message
                    value = program_message.value
                    self.set_program(channel,value)
                    print(f"MIDI track {track_id} - {track_name}: Channel {channel} Program {value}")
                    self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                        f"MIDI track {track_id} - {track_name}: Channel {channel} Program {value}"),None))
                elif command == 0xd0:
                    # channel pressure
                    channel_pressure_message:ChannelPressureMessage = event.message
                    value = channel_pressure_message.value
                    self.set_pressure(channel,value)
                    print(f"MIDI track {track_id} - {track_name}: Channel {channel} Pressure {value}")
                    self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                        f"MIDI track {track_id} - {track_name}: Channel {channel} Pressure {value}"),None))
                elif command == 0xe0:
                    # pitch bend change
                    pitch_bend_message:PitchBendMessage = event.message
                    value = pitch_bend_message.value
                    self.set_pitch_bend(channel,value)
                    print(f"MIDI track {track_id} - {track_name}: Channel {channel} Pitch Bend {value}")
                    self.observer.put((MODULE_LOG,(self.mod_id,"debug",
                        f"MIDI track {track_id} - {track_name}: Channel {channel} Pitch Bend {value}"),None))
                else:
                    print("",end="\r")
                    #print("System Exclusive MIDI Event",event)
            else:
                print("Unknown event",event)

        
        for track_id in list(tracks_to_remove):
            self.midi_tracks.pop(track_id)
            self.midi_track_idx.pop(track_id)
            self.midi_track_instrument_name.pop(track_id)
            self.midi_track_name.pop(track_id)
            self.midi_track_time.pop(track_id)
    
        # compute any remaining chunk data
        if self.current_chunk_idx < self.chunk_size:
            self.compute_chunk_data(self.chunk_size)

        #print(self.current_tick,self.current_chunk)

        for ch in range(self.midi_channels):
            self.output_buffer[ch].append(self.chunk_output[ch])
        self.current_chunk+=1
        return AM_CONTINUE

    def open_midi_file(self):
        if len(self.filename)==0:
            return
        try:
            c=midiparser.MIDIFile(self.filename)
            c.parse()
            #print(str(c))
            if c.division >= 32786:
                raise Exception("Subdivisions of second is not supported.")
            self.midi_ticks_per_quarter = c.division
            #print("ticks per quarter",self.midi_ticks_per_quarter)
            self.observer.put((MODULE_LOG,(self.mod_id,"info",f"MIDI ticks per quarter {c.division}"), None))
            self.set_time_params()
            self.all_midi_events=[]
            for idx, track in enumerate(c):
                self.midi_tracks[idx]=track
                self.midi_track_idx[idx]=0
                self.midi_track_name[idx]=""
                self.midi_track_instrument_name[idx]=""
                self.midi_track_time[idx]=0
                track.parse()
                self.observer.put((MODULE_LOG,(self.mod_id,"info",f"MIDI preparing track {idx}"), None))
                for event_idx in range(len(track)):
                    event=track[event_idx]
                    self.all_midi_events.append((event.time,idx,event_idx))
            self.observer.put((MODULE_LOG,(self.mod_id,"info",f"MIDI found {len(self.all_midi_events)} events"), None))        
            self.all_midi_events.sort(key=lambda x:(x[0],x[1],x[2]))
            self.ready = True
        except Exception as e:
            self.observer.put(
                    (MODULE_ERROR, (self.mod_id, f"{self.filename} could not be opened. {e}."), None))
            self.ready = False

    def get_widget_params(self):
        return super().get_widget_params() | {
            'meta_order': ['model_dir','midi_file'],
            'provide_update_button': True,
            'model_dir': {
                'name': 'Model Directory',
                'value': self.model_dir,
                'type': 'read-dirname',
                'instant_update': False
            },
            'midi_file':{
                'name': 'MIDI File',
                'value': self.filename,
                'type': 'read-filename',
                'filetypes': [("MIDI files","*.mid *.MID *.midi *.MIDI *.Midi"),("All files","*.*")],
                'instant_update': False
            }
        }

    def set_widget_params(self, params):
        super().set_widget_params(params)

        self.close()
        self.filename = params['midi_file']['value']
        self.model_dir = params['model_dir']['value']
        self.open()

    def get_status(self):
        if self.ready:
            status = {
                'topleft':Path(self.filename).name,
            }
        else:
            status = {
                'bottom':"Valid MIDI filename required."
            }
        return status

def midi_kernel_process(from_mod:Queue,
            to_mod:Queue,
            chunks:Queue,
            filename:str="",
            model_dir:str="",
            midi_channels:int=16,
            cpus:int=4,
            sample_rate:int=AV_RATE,
            chunk_size:int=1024,
            dtype=np.float32,
            observer:Queue=None,
            mod_id:ModId=None,
            **kwargs):
    midi_kernel = MidiKernel(filename,
        model_dir,
        midi_channels,
        cpus,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        dtype=dtype,
        observer=observer,
        mod_id=mod_id,
        polled=True)
    done=False
    running=False
    print("midi kernel process starting loop")
    while True:
        while (not done) and from_mod.empty():
            if not chunks.full() and running:
                ret=midi_kernel.process_next_chunk()
                if ret==AM_ERROR or ret==AM_COMPLETED:
                    chunks.put((None,ret))
                    print("midi kernel finished with code",ret)
                    done=True
                else:
                    outputs=[]
                    for ch in range(midi_channels):
                        outputs.append(midi_kernel.output_buffer[ch].pop(0))
                    chunks.put((outputs,AM_CONTINUE))
                    if chunks.qsize() < 100:
                        print("WARNING: buffer is low: ",chunks.qsize())
            else:
                time.sleep(0.0001)
        
        if not from_mod.empty():
            print("midi kernel process getting command")
            command,data = from_mod.get()
            print("got command",command,data)
            if command=="quit":
                break
            elif command=="get_widget_params":
                to_mod.put(("RET_get_widget_params",midi_kernel.get_widget_params()))
                print("sent RET get widget params")
            elif command=="set_widget_params":
                to_mod.put(("RET_set_widget_params",midi_kernel.set_widget_params(data)))
            elif command=='start':
                to_mod.put(("RET_start",midi_kernel.start()))
                running=True
            elif command=='stop':
                to_mod.put(("RET_stop",midi_kernel.stop()))
                running=False
            elif command=='open':
                to_mod.put(("RET_open",midi_kernel.open()))
            elif command=='close':
                to_mod.put(("RET_close",midi_kernel.close()))
            elif command=='reset':
                to_mod.put(("RET_reset",midi_kernel.reset()))
                running=False
            elif command=='get_status':
                to_mod.put(("RET_get_status",midi_kernel.get_status()))
        else:
            time.sleep(0.0001)
    print("midi kernel process terminating")
        
        
@audiomod
class Midi(AudioModule):
    """ Read a midi file and output synthesized sounds. """

    name = "MIDI"
    category = "Musical"
    description = ("Basic MIDI support (MIDI 1.0). 16 MIDI channels (module outputs): "
        "3 channels (left, right, center) per module output.")

    def __init__(self,filename:str="",
            model_dir:str="",
            midi_channels:int=16,
            cpus:int=4,
            chunk_queue_size:int=4096,
            **kwargs):
        super().__init__(num_inputs=0,
            num_outputs=midi_channels,
            out_chs=[3]*midi_channels,
            **kwargs)
        self.filename=filename
        self.model_dir=model_dir
        self.midi_channels=midi_channels
        self.cpus=cpus

        self.chunk_queue_size=chunk_queue_size
        self.manager=Manager()
        self.to_kernel = self.manager.Queue()
        self.from_kernel = self.manager.Queue()
        self.chunks = self.manager.Queue(self.chunk_queue_size)

        self.midi_kernel = Process(target = midi_kernel_process,args=(self.to_kernel,
            self.from_kernel,
            self.chunks,
            self.filename,
            self.model_dir,
            self.midi_channels,
            self.cpus,
            self.sample_rate,
            self.chunk_size,
            self.dtype,
            self.observer,
            self.mod_id))
        print("starting midi kernel process")
        self.midi_kernel.start()
        print("midi kernel process started")
        self.running=True

    def get_widget_params(self):
        self.to_kernel.put(("get_widget_params",None))
        _,ret = self.from_kernel.get()
        return ret

    def set_widget_params(self, params):
        self.to_kernel.put(("set_widget_params",params))
        print("waiting for response to set widget params")
        _,ret = self.from_kernel.get()
        return ret

    def start(self):
        self.to_kernel.put(("start",None))
        _,ret = self.from_kernel.get()
        time.sleep(5)
        return ret

    def stop(self):
        self.to_kernel.put(("stop",None))
        _,ret = self.from_kernel.get()
        return ret

    def open(self):
        if not self.running:
            self.to_kernel = self.manager.Queue()
            self.from_kernel = self.manager.Queue()
            self.chunks = self.manager.Queue(self.chunk_queue_size)
            self.midi_kernel = Process(target = midi_kernel_process,args=(self.to_kernel,
                self.from_kernel,
                self.chunks,
                self.filename,
                self.model_dir,
                self.midi_channels,
                self.cpus,
                self.sample_rate,
                self.chunk_size,
                self.dtype,
                self.observer,
                self.mod_id))
            self.midi_kernel.start()
            self.running=True
        self.to_kernel.put(("open",None))
        _,ret = self.from_kernel.get()
        return ret

    def close(self):
        self.to_kernel.put(("close",None))
        _,ret = self.from_kernel.get()
        self.to_kernel.put(("quit",None))
        print("Joining with midi kernel process.")
        self.midi_kernel.join()
        self.midi_kernel.close()
        self.running=False
        return ret

    def reset(self):
        self.to_kernel.put(("reset",None))
        _,ret = self.from_kernel.get()
        return ret

    async def next_chunk(self) -> AM_COMPLETED | AM_CONTINUE | AM_INPUT_REQUIRED | AM_ERROR | AM_CYCLIC_UNDERRUN:
        # will block if the kernel is falling behind
        chunk,retcode = self.chunks.get()
        if chunk==None:
            return retcode
        else:
            for ch in range(self.num_outputs):
                self.send_signal(chunk[ch],ch)
            return retcode

    def get_status(self):
        self.to_kernel.put(("get_status",None))
        _,ret = self.from_kernel.get()
        return ret
    