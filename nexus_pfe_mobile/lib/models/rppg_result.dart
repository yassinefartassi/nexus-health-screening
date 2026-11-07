import 'dart:ui';

import '../services/rppg_algorithms.dart' show HrvResult;

/// One rPPG estimation cycle's output: both CHROM and POS are computed side
/// by side (rather than picking one) so the two can be visually compared,
/// per the thesis's original evaluation goal on desktop.
class RppgResult {
  final double bpmChrom;
  final double bpmPos;
  final List<double> chromWaveform;
  final List<double> posWaveform;

  /// Heart-rate variability, computed from the POS waveform (POS is the
  /// desktop prototype's default algorithm).
  final HrvResult hrv;

  /// Effective sampling rate (frames actually buffered per second), used by
  /// both the bandpass filter and the FFT peak search.
  final double fps;

  final Rect? faceBox;
  final Size? imageSize;

  const RppgResult({
    required this.bpmChrom,
    required this.bpmPos,
    required this.chromWaveform,
    required this.posWaveform,
    required this.hrv,
    required this.fps,
    required this.faceBox,
    required this.imageSize,
  });
}
