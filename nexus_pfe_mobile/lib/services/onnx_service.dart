// Generic ONNX Runtime wrapper shared by every on-device model in this app
// (ocular classifier, dermatology classifier, and any future model).
//
// Design mirrors the desktop prototype's `common_onnx.py`: instead of
// hard-coding an input resolution, we read the expected input shape from the
// ONNX graph itself at load time. This lets ocular_inference.dart and
// dermatology_inference.dart share one code path even though the two models
// were exported independently and may not share the same input size.
//
// All inference runs fully on-device via the ONNX Runtime Android/iOS
// bindings (flutter_onnxruntime) — no network call is made at any point.
import 'dart:developer' as developer;
import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// Resolved shape info for a model's single image input tensor.
class OnnxInputSpec {
  final int height;
  final int width;
  final int channels;

  /// true  -> tensor layout is NCHW (channels immediately after batch dim)
  /// false -> tensor layout is NHWC (channels last)
  final bool channelsFirst;

  const OnnxInputSpec({
    required this.height,
    required this.width,
    required this.channels,
    required this.channelsFirst,
  });

  List<int> get nchwShape => [1, channels, height, width];
  List<int> get nhwcShape => [1, height, width, channels];

  List<int> get tensorShape => channelsFirst ? nchwShape : nhwcShape;
}

/// Thrown when a model asset fails to load or produces an unexpected shape.
class OnnxLoadException implements Exception {
  final String message;
  final Object? cause;
  final StackTrace? causeStackTrace;
  OnnxLoadException(this.message, [this.cause, this.causeStackTrace]);

  /// Full detail including the wrapped platform/runtime exception — always
  /// use this (not [message] alone) when surfacing the error to logs or UI,
  /// otherwise the actual root cause (missing asset, bad opset, corrupt
  /// weights, etc.) gets silently dropped and only the generic wrapper text
  /// is shown.
  @override
  String toString() =>
      'OnnxLoadException: $message${cause != null ? '\nCaused by: $cause' : ''}';
}

class OnnxService {
  final String assetPath;

  OrtSession? _session;
  String? _inputName;
  String? _outputName;
  OnnxInputSpec? _inputSpec;

  OnnxService({required this.assetPath});

  bool get isLoaded => _session != null;
  OnnxInputSpec get inputSpec {
    final spec = _inputSpec;
    if (spec == null) {
      throw StateError('OnnxService.load() must complete before use.');
    }
    return spec;
  }

  /// Loads the model from the Flutter asset bundle and resolves its input
  /// tensor shape. Safe to call once; call [dispose] before reloading.
  Future<void> load() async {
    if (_session != null) return;

    final OrtSession session;
    try {
      session = await OnnxRuntime().createSessionFromAsset(assetPath);
    } catch (e, st) {
      // Log the real underlying exception (missing asset, unsupported op,
      // bad opset, corrupt/external-data weights, etc.) — without this the
      // UI and logs only ever show the generic wrapper message below and
      // the actual cause is impossible to diagnose from a device report.
      developer.log(
        'Failed to load ONNX model at $assetPath',
        name: 'OnnxService',
        error: e,
        stackTrace: st,
      );
      throw OnnxLoadException(
        'Failed to load ONNX model at $assetPath',
        e,
        st,
      );
    }

    if (session.inputNames.isEmpty || session.outputNames.isEmpty) {
      throw OnnxLoadException(
        '$assetPath has no input/output tensors — is this a valid exported classifier?',
      );
    }

    _session = session;
    _inputName = session.inputNames.first;
    _outputName = session.outputNames.first;
    _inputSpec = await _resolveInputSpec(session, _inputName!);
  }

  /// Reads the input tensor's declared shape from the ONNX graph and infers
  /// (height, width, channels, layout). Falls back to the common default of
  /// 224x224 RGB / NCHW if the shape can't be introspected on this platform
  /// (iOS does not currently expose getInputInfo()) or the dims are dynamic.
  Future<OnnxInputSpec> _resolveInputSpec(
    OrtSession session,
    String inputName,
  ) async {
    List<dynamic>? rawShape;
    try {
      final inputInfo = await session.getInputInfo();
      final match = inputInfo.firstWhere(
        (info) => info['name'] == inputName,
        orElse: () => inputInfo.first,
      );
      rawShape = match['shape'] as List<dynamic>?;
    } catch (_) {
      // getInputInfo() is unsupported on some platforms (e.g. iOS) — fall
      // through to the default below, same as the Python prototype does.
    }

    if (rawShape == null || rawShape.length != 4) {
      return const OnnxInputSpec(
        height: 224,
        width: 224,
        channels: 3,
        channelsFirst: true,
      );
    }

    final dims = rawShape
        .map((d) => (d is int && d > 0) ? d : null)
        .toList(growable: false);

    // NCHW: [N, 3, H, W]
    if (dims[1] == 1 || dims[1] == 3) {
      return OnnxInputSpec(
        height: dims[2] ?? 224,
        width: dims[3] ?? 224,
        channels: dims[1]!,
        channelsFirst: true,
      );
    }
    // NHWC: [N, H, W, 3]
    if (dims[3] == 1 || dims[3] == 3) {
      return OnnxInputSpec(
        height: dims[1] ?? 224,
        width: dims[2] ?? 224,
        channels: dims[3]!,
        channelsFirst: false,
      );
    }

    return const OnnxInputSpec(
      height: 224,
      width: 224,
      channels: 3,
      channelsFirst: true,
    );
  }

  /// Runs inference on a preprocessed [Float32List] tensor (already resized,
  /// normalized, and laid out to match [inputSpec]) and returns the raw
  /// flattened output (logits or probabilities — the caller decides which).
  Future<List<double>> run(Float32List input, List<int> shape) async {
    final session = _session;
    final inputName = _inputName;
    final outputName = _outputName;
    if (session == null || inputName == null || outputName == null) {
      throw StateError('OnnxService.load() must complete before run().');
    }

    final inputTensor = await OrtValue.fromList(input, shape);
    try {
      final outputs = await session.run({inputName: inputTensor});
      try {
        final outputTensor = outputs[outputName];
        if (outputTensor == null) {
          throw OnnxLoadException(
            'Model $assetPath did not produce output "$outputName"',
          );
        }
        final flat = await outputTensor.asFlattenedList();
        return flat.map((v) => (v as num).toDouble()).toList(growable: false);
      } finally {
        for (final tensor in outputs.values) {
          await tensor.dispose();
        }
      }
    } finally {
      await inputTensor.dispose();
    }
  }

  Future<void> dispose() async {
    await _session?.close();
    _session = null;
    _inputName = null;
    _outputName = null;
    _inputSpec = null;
  }
}
