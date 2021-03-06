# Module Library

A set of modules that use the `audiomodule` base class.

## Downsample

Remove every `k`-th sample from the input, where `k` is an integer. If `k=2` the module outputs half as many samples as it inputs.

## Upsample

Insert `k-1` zeroes between samples, where `k` is an integer. If `k=2` the module outputs twice as many samples as it inputs. If `sample_and_hold` is `True` then sample and hold (replication) is used instead of zeroes.

## Resample

Make use of `Downsample` and `Upsample` to resample the input by a real factor. A factor of 2 outputs twice as many samples as input. The real factor is converted to a rational approximation and sub-sample correction is applied to eliminate accumulation error.

## Stretch

Use phase vocoding to stretch the input audio without altering its pitch.

## Pitch Shift

Change the pitch of the input audio, in terms of semitones, using `Stretch` and `Resample`.

## IIR Design

Implements an interface to SciPy's IIR filter design.

## Frequency Modulation

Modulate a carrier frequency with a signal.

## To Wave File

Write the input data to a WAV file.

## Wave Generator

Generate a sinusoidal wave.

## Whitenoise Generator

Generate white noise.

## Delay

Introduce a delay to the signal.

## Audio Out

Play audio on the system's audio device.

## Channel Combine

Combine multiple single-channel inputs into a single multi-channel output. For example, to combine a left channel and a right channel into a stero audio signal.

## Channel Split

Select channels to output from a multi-channel input.

## Linear

Linearly combine multiple inputs to a single output.

## Mix Down

Mix all channels to a single channel.

## Multiply

Multiply two inputs together.

## Normalize

Normalize the input using a running maximum.

## From WAV File

Read data from a WAV file.

## MIDI

Reads a MIDI file and produces synthesized music on the output.

## MIDI Support

Not a module itself, but provides support functionality for using one-shot sound samples in a MIDI module, as well as other music related definitions.
