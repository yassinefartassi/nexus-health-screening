// Live rPPG pipeline: face ROI tracking + RGB buffering + CHROM/POS/BPM/HRV.
//
// The screen owns the CameraController and pushes each streamed frame here
// via [processFrame]; this service owns everything vision/signal-processing
// related — face detection, ROI extraction, the rolling RGB buffer, and
// orchestrating rppg_algorithms.dart.
//
// Frame format: the screen must request `ImageFormatGroup.nv21` on Android
// and the (iOS default) `bgra8888` on iOS so each CameraImage arrives as a
// single densely-packed plane — this keeps the pixel-sampling code below
// simple and avoids a full-frame YUV->RGB conversion every frame.
import 'dart:async' show unawaited;
import 'dart:collection';
import 'dart:io';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../models/rppg_result.dart';
import 'rppg_algorithms.dart';

class _RgbSample {
  final int timestampMs;
  final double r;
  final double g;
  final double b;
  const _RgbSample(this.timestampMs, this.r, this.g, this.b);
}

class RppgService {
  static const int bufferSeconds = 10;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(performanceMode: FaceDetectorMode.fast),
  );

  final Queue<_RgbSample> _buffer = Queue<_RgbSample>();

  int _frameCounter = 0;
  Rect? _lastFaceBox;
  Size? _lastImageSize;
  bool _detecting = false;

  bool get faceFound => _lastFaceBox != null;
  Rect? get lastFaceBox => _lastFaceBox;
  Size? get lastImageSize => _lastImageSize;

  double get bufferFillFraction {
    if (_buffer.isEmpty) return 0.0;
    final spanSec =
        (_buffer.last.timestampMs - _buffer.first.timestampMs) / 1000.0;
    return (spanSec / bufferSeconds).clamp(0.0, 1.0);
  }

  /// Feeds one camera frame through face detection (throttled) + ROI RGB
  /// extraction (every accepted frame) into the rolling buffer.
  Future<void> processFrame({
    required CameraImage image,
    required InputImageRotation rotation,
  }) async {
    _frameCounter++;
    _lastImageSize = Size(image.width.toDouble(), image.height.toDouble());

    // Face detection is the expensive step; run it every other frame (and
    // whenever we don't currently have a box) and reuse the last known box
    // in between so RGB sampling — which drives the fps estimate CHROM/POS
    // depend on — stays close to the camera's native frame rate.
    if (!_detecting && (_lastFaceBox == null || _frameCounter.isEven)) {
      _detecting = true;
      unawaited(_detectFace(image, rotation).then((box) {
        _lastFaceBox = box;
        _detecting = false;
      }));
    }

    final faceBox = _lastFaceBox;
    if (faceBox == null) return;

    final roi = _foreheadRoi(faceBox, image.width, image.height);
    if (roi.width < 2 || roi.height < 2) return;

    final mean = _meanRgbInRoi(image, roi);
    if (mean == null) return;

    final now = DateTime.now().millisecondsSinceEpoch;
    _buffer.add(_RgbSample(now, mean.$1, mean.$2, mean.$3));
    _trimBuffer(now);
  }

  Future<Rect?> _detectFace(CameraImage image, InputImageRotation rotation) async {
    final inputImage = _buildInputImage(image, rotation);
    if (inputImage == null) return null;
    try {
      final faces = await _faceDetector.processImage(inputImage);
      if (faces.isEmpty) return null;
      faces.sort((a, b) => (b.boundingBox.width * b.boundingBox.height)
          .compareTo(a.boundingBox.width * a.boundingBox.height));
      return faces.first.boundingBox;
    } catch (_) {
      return null;
    }
  }

  InputImage? _buildInputImage(CameraImage image, InputImageRotation rotation) {
    if (image.planes.length != 1) return null;
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) return null;
    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  /// Forehead band within the detected face box: 30-70% of width, 8-25% of
  /// height (upper-face region, centered, avoiding hair/eyebrows) — the same
  /// ROI geometry validated in the desktop prototype's `rppg/processor.py`.
  Rect _foreheadRoi(Rect face, int imageWidth, int imageHeight) {
    final fx1 = face.left + 0.30 * face.width;
    final fx2 = face.left + 0.70 * face.width;
    final fy1 = face.top + 0.08 * face.height;
    final fy2 = face.top + 0.25 * face.height;
    return Rect.fromLTRB(fx1, fy1, fx2, fy2)
        .intersect(Rect.fromLTWH(0, 0, imageWidth.toDouble(), imageHeight.toDouble()));
  }

  (double, double, double)? _meanRgbInRoi(CameraImage image, Rect roi) {
    if (Platform.isAndroid) return _meanRgbNv21(image, roi);
    if (Platform.isIOS) return _meanRgbBgra8888(image, roi);
    return null;
  }

  /// Android: single NV21 plane (Y, then interleaved V/U at half resolution).
  (double, double, double)? _meanRgbNv21(CameraImage image, Rect roi) {
    final bytes = image.planes[0].bytes;
    final w = image.width;
    final ySize = w * image.height;
    final x0 = roi.left.floor(), x1 = roi.right.floor();
    final y0 = roi.top.floor(), y1 = roi.bottom.floor();

    double sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    // Stride 2 in both axes: halves the pixel count for a ~2x speedup with
    // negligible effect on a spatial mean over a several-thousand-pixel ROI.
    for (int y = y0; y < y1; y += 2) {
      final yRowBase = y * w;
      final uvRowBase = ySize + (y ~/ 2) * w;
      for (int x = x0; x < x1; x += 2) {
        final yIndex = yRowBase + x;
        final uvIndex = uvRowBase + (x ~/ 2) * 2;
        if (yIndex >= bytes.length || uvIndex + 1 >= bytes.length) continue;

        final yy = bytes[yIndex].toDouble();
        final v = bytes[uvIndex].toDouble();
        final u = bytes[uvIndex + 1].toDouble();

        // BT.601 YUV -> RGB.
        final r = yy + 1.402 * (v - 128);
        final g = yy - 0.344136 * (u - 128) - 0.714136 * (v - 128);
        final b = yy + 1.772 * (u - 128);
        sumR += r.clamp(0, 255);
        sumG += g.clamp(0, 255);
        sumB += b.clamp(0, 255);
        count++;
      }
    }
    if (count == 0) return null;
    return (sumR / count, sumG / count, sumB / count);
  }

  /// iOS: single BGRA8888 plane.
  (double, double, double)? _meanRgbBgra8888(CameraImage image, Rect roi) {
    final bytes = image.planes[0].bytes;
    final bytesPerRow = image.planes[0].bytesPerRow;
    final x0 = roi.left.floor(), x1 = roi.right.floor();
    final y0 = roi.top.floor(), y1 = roi.bottom.floor();

    double sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    for (int y = y0; y < y1; y += 2) {
      final rowBase = y * bytesPerRow;
      for (int x = x0; x < x1; x += 2) {
        final index = rowBase + x * 4;
        if (index + 2 >= bytes.length) continue;
        sumB += bytes[index];
        sumG += bytes[index + 1];
        sumR += bytes[index + 2];
        count++;
      }
    }
    if (count == 0) return null;
    return (sumR / count, sumG / count, sumB / count);
  }

  void _trimBuffer(int nowMs) {
    while (_buffer.isNotEmpty &&
        (nowMs - _buffer.first.timestampMs) > bufferSeconds * 1000) {
      _buffer.removeFirst();
    }
  }

  /// Runs CHROM + POS on the current buffer. Returns null until there's
  /// enough buffered data (~2s minimum, same guard as the Python prototype).
  RppgResult? computeResult() {
    final n = _buffer.length;
    if (n < 2) return null;

    final samples = _buffer.toList(growable: false);
    final durationSec =
        (samples.last.timestampMs - samples.first.timestampMs) / 1000.0;
    final fps = durationSec > 0 ? n / durationSec : 0.0;
    if (fps < 5) return null; // not enough temporal resolution yet

    final rgb = samples.map((s) => [s.r, s.g, s.b]).toList(growable: false);

    final chromRaw = chromSignal(rgb);
    final posRaw = posSignal(rgb);
    final chromFiltered = bandpassFilter(chromRaw, fps);
    final posFiltered = bandpassFilter(posRaw, fps);

    if (chromFiltered.length < fps * 2) return null;

    final chromBpm = estimateBpm(chromFiltered, fps);
    final posBpm = estimateBpm(posFiltered, fps);
    // HRV is computed from the POS waveform — POS is the desktop
    // prototype's default algorithm (config.RPPG_DEFAULT_ALGO = "POS").
    final hrv = computeHrv(posFiltered, fps);

    return RppgResult(
      bpmChrom: chromBpm,
      bpmPos: posBpm,
      chromWaveform: chromFiltered,
      posWaveform: posFiltered,
      hrv: hrv,
      fps: fps,
      faceBox: _lastFaceBox,
      imageSize: _lastImageSize,
    );
  }

  void reset() {
    _buffer.clear();
    _lastFaceBox = null;
  }

  Future<void> dispose() async {
    await _faceDetector.close();
  }
}
