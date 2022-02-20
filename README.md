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

