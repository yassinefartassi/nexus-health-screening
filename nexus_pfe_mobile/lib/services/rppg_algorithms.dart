// Remote photoplethysmography (rPPG) signal processing.
//
// Original Dart re-implementation of two published, well-validated
// chrominance-based rPPG methods:
//
//   - CHROM: de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based
//            rPPG", IEEE Trans. Biomed. Eng., 2013.
//   - POS:   Wang et al., "Algorithmic Principles of Remote-PPG",
//            IEEE Trans. Biomed. Eng., 2016 (Plane-Orthogonal-to-Skin).
//
// The linear-combination / projection math below was cross-checked against
// this project's own Python prototype (NEXUS_PFE / health_ai_suite,
// `rppg/algorithms.py`), which in turn used rPPG-Toolbox
// (https://github.com/ubicomplab/rPPG-Toolbox, Responsible AI License v1.1)
// as the reference implementation during prototyping on desktop/Ubuntu. No
// rPPG-Toolbox code is vendored or translated here — this file only
// reimplements the published algorithms' equations directly in Dart. See
// README.md for the full citation.
import 'dart:math';
import 'dart:typed_data';

import 'package:fftea/fftea.dart';

/// Plausible resting/active human heart rate range used both to constrain
/// the bandpass filter and to restrict the FFT peak search.
const double kMinBpm = 42;
const double kMaxBpm = 180;

/// Given a buffer of mean RGB samples (one per frame, R/G/B in that order,
/// on any consistent scale e.g. 0-255), returns the CHROM pulse signal.
List<double> chromSignal(List<List<double>> rgbSamples) {
  final n = rgbSamples.length;
  final means = _channelMeans(rgbSamples);

  final xs = List<double>.filled(n, 0);
  final ys = List<double>.filled(n, 0);
  for (int i = 0; i < n; i++) {
    final rn = rgbSamples[i][0] / (means[0] + 1e-8);
    final gn = rgbSamples[i][1] / (means[1] + 1e-8);
    final bn = rgbSamples[i][2] / (means[2] + 1e-8);
    xs[i] = 3 * rn - 2 * gn;
    ys[i] = 1.5 * rn + gn - 1.5 * bn;
  }

  final alpha = _std(xs) / (_std(ys) + 1e-8);
  final signal = List<double>.generate(n, (i) => xs[i] - alpha * ys[i]);
  return _linearDetrend(signal);
}

/// Given a buffer of mean RGB samples, returns the POS (Plane-Orthogonal-to-
/// Skin) pulse signal.
List<double> posSignal(List<List<double>> rgbSamples) {
  final n = rgbSamples.length;
  final means = _channelMeans(rgbSamples);

  final s1 = List<double>.filled(n, 0);
  final s2 = List<double>.filled(n, 0);
  for (int i = 0; i < n; i++) {
    final rn = rgbSamples[i][0] / (means[0] + 1e-8);
    final gn = rgbSamples[i][1] / (means[1] + 1e-8);
    final bn = rgbSamples[i][2] / (means[2] + 1e-8);
    s1[i] = gn - bn;
    s2[i] = gn + bn - 2 * rn;
  }

  final alpha = _std(s1) / (_std(s2) + 1e-8);
  final signal = List<double>.generate(n, (i) => s1[i] + alpha * s2[i]);
  return _linearDetrend(signal);
}

List<double> _channelMeans(List<List<double>> rgbSamples) {
  double r = 0, g = 0, b = 0;
  for (final s in rgbSamples) {
    r += s[0];
    g += s[1];
    b += s[2];
  }
  final n = rgbSamples.length;
  return [r / n, g / n, b / n];
}

double _std(List<double> x) {
  final mean = x.reduce((a, b) => a + b) / x.length;
  final variance =
      x.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / x.length;
  return sqrt(variance);
}

/// Removes the best-fit linear trend (least-squares) from [x], matching
/// `scipy.signal.detrend(x)`'s default `type='linear'` behavior used by the
/// Python prototype.
List<double> _linearDetrend(List<double> x) {
  final n = x.length;
  if (n < 2) return List<double>.from(x);

  double sumT = 0, sumY = 0, sumTT = 0, sumTY = 0;
  for (int i = 0; i < n; i++) {
    sumT += i;
    sumY += x[i];
    sumTT += i * i;
    sumTY += i * x[i];
  }
  final denom = n * sumTT - sumT * sumT;
  final slope = denom != 0 ? (n * sumTY - sumT * sumY) / denom : 0.0;
  final intercept = (sumY - slope * sumT) / n;

  return List<double>.generate(n, (i) => x[i] - (intercept + slope * i));
}

/// Zero-phase bandpass filter restricted to plausible heart-rate
/// frequencies (0.7-4 Hz, i.e. 42-240 BPM, intersected with [kMinBpm,
/// kMaxBpm] by the caller's FFT search).
///
/// Implemented as a single RBJ-cookbook constant-skirt-gain biquad
/// (https://www.w3.org/TR/audio-eq-cookbook/), applied forward then
/// backward (classic "filtfilt" trick) to cancel phase distortion. The
/// Python prototype instead used `scipy.signal.butter(3, ..., sos=True)` +
/// `sosfiltfilt`; this is a deliberately lighter-weight zero-phase bandpass
/// with the same intent (bandlimit the raw CHROM/POS signal before the FFT
/// peak search) rather than a numerically identical port — there is no
/// SciPy-equivalent filter design routine in pure Dart.
List<double> bandpassFilter(
  List<double> signal,
  double fps, {
  double lowHz = 0.7,
  double highHz = 4.0,
}) {
  List<double> applyOnce(List<double> x) {
    final f0 = sqrt(lowHz * highHz);
    final bandwidth = highHz - lowHz;
    final q = f0 / bandwidth;
    final w0 = 2 * pi * f0 / fps;
    final alpha = sin(w0) / (2 * q);
    final a0 = 1 + alpha;
    final b0 = alpha / a0;
    final b2 = -alpha / a0;
    final a1 = (-2 * cos(w0)) / a0;
    final a2 = (1 - alpha) / a0;

    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    final out = List<double>.filled(x.length, 0);
    for (int i = 0; i < x.length; i++) {
      final xi = x[i];
      final yi = b0 * xi + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = xi;
      y2 = y1;
      y1 = yi;
      out[i] = yi;
    }
    return out;
  }

  final forward = applyOnce(signal);
  final backward = applyOnce(forward.reversed.toList());
  return backward.reversed.toList();
}

/// Estimates BPM from a bandpass-filtered pulse signal via FFT peak search,
/// matching the Hann-window + narrowed-search-band approach in the Python
/// prototype (guards against locking onto filter edge/ringing artifacts).
double estimateBpm(
  List<double> filteredSignal,
  double fps, {
  double minBpm = kMinBpm,
  double maxBpm = kMaxBpm,
}) {
  final n = filteredSignal.length;
  if (n < fps * 2) return 0.0; // need at least ~2s of data

  final windowed = List<double>.generate(
    n,
    (i) => filteredSignal[i] * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))),
  );

  final fftSize = _nextPowerOfTwo(n);
  final padded = Float64List(fftSize);
  for (int i = 0; i < n; i++) {
    padded[i] = windowed[i];
  }

  final fft = FFT(fftSize);
  final spectrum = fft.realFft(padded).discardConjugates().magnitudes();
  final freqResolution = fps / fftSize;

  const searchMarginBpm = 6.0; // avoids Butterworth-edge-style ringing near cutoffs
  double bestBpm = 0.0;
  double bestPower = -1;
  bool foundInMargin = false;

  for (int i = 0; i < spectrum.length; i++) {
    final bpm = (i * freqResolution) * 60;
    if (bpm >= minBpm + searchMarginBpm && bpm <= maxBpm - searchMarginBpm) {
      foundInMargin = true;
      final power = spectrum[i] * spectrum[i];
      if (power > bestPower) {
        bestPower = power;
        bestBpm = bpm;
      }
    }
  }

  if (!foundInMargin) {
    for (int i = 0; i < spectrum.length; i++) {
      final bpm = (i * freqResolution) * 60;
      if (bpm >= minBpm && bpm <= maxBpm) {
        final power = spectrum[i] * spectrum[i];
        if (power > bestPower) {
          bestPower = power;
          bestBpm = bpm;
        }
      }
    }
  }

  return bestBpm;
}

int _nextPowerOfTwo(int n) {
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

/// Basic time-domain heart-rate-variability metrics computed from
/// peak-to-peak (beat-to-beat) intervals in a filtered pulse waveform.
class HrvResult {
  /// Standard deviation of NN (beat-to-beat) intervals, in milliseconds.
  final double sdnnMs;

  /// Root mean square of successive NN interval differences, in
  /// milliseconds.
  final double rmssdMs;

  final int beatsDetected;

  const HrvResult({
    required this.sdnnMs,
    required this.rmssdMs,
    required this.beatsDetected,
  });

  static const empty = HrvResult(sdnnMs: 0, rmssdMs: 0, beatsDetected: 0);
}

/// Detects peaks in [filteredSignal] (simple local-maximum search with a
/// minimum-distance constraint derived from [maxBpm], so noise can't
/// register as extra beats) and computes SDNN / RMSSD from the resulting
/// inter-beat intervals.
HrvResult computeHrv(
  List<double> filteredSignal,
  double fps, {
  double maxBpm = kMaxBpm,
}) {
  final minDistanceSamples = ((fps * 60.0) / maxBpm)
      .round()
      .clamp(1, filteredSignal.isEmpty ? 1 : filteredSignal.length);

  final peaks = <int>[];
  for (int i = 1; i < filteredSignal.length - 1; i++) {
    final isLocalMax = filteredSignal[i] > filteredSignal[i - 1] &&
        filteredSignal[i] >= filteredSignal[i + 1];
    if (!isLocalMax) continue;

    if (peaks.isEmpty || (i - peaks.last) >= minDistanceSamples) {
      peaks.add(i);
    } else if (filteredSignal[i] > filteredSignal[peaks.last]) {
      // Two candidate peaks too close together: keep the taller one.
      peaks[peaks.length - 1] = i;
    }
  }

  if (peaks.length < 3) {
    return HrvResult(sdnnMs: 0, rmssdMs: 0, beatsDetected: peaks.length);
  }

  final ibiMs = <double>[
    for (int i = 1; i < peaks.length; i++)
      (peaks[i] - peaks[i - 1]) * (1000.0 / fps),
  ];

  final meanIbi = ibiMs.reduce((a, b) => a + b) / ibiMs.length;
  final sdnn = sqrt(
    ibiMs.map((v) => (v - meanIbi) * (v - meanIbi)).reduce((a, b) => a + b) /
        ibiMs.length,
  );

  double sumSquaredDiffs = 0;
  for (int i = 1; i < ibiMs.length; i++) {
    final diff = ibiMs[i] - ibiMs[i - 1];
    sumSquaredDiffs += diff * diff;
  }
  final rmssd = ibiMs.length > 1
      ? sqrt(sumSquaredDiffs / (ibiMs.length - 1))
      : 0.0;

  return HrvResult(sdnnMs: sdnn, rmssdMs: rmssd, beatsDetected: peaks.length);
}
