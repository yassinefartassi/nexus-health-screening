// Dermatology (DermNet-derived) classifier: 10 curated non-cancerous skin
// condition categories + "normal" = 11 classes, EfficientNetV2-S backbone,
// ~61.6% test accuracy (see dermatho-assets/classification_report.txt).
//
// This bundle was re-exported from the canonical dermatho-assets/dermnet_v2s_model.keras
// (2026-07-12 hardening pass) after an audit found this app had been shipping a
// stale, unrelated 24-class EfficientNetB0 model. `assets/models/dermnet_class_names.json`
// now carries the verified 11-class order verbatim from dermatho-assets/dermnet_class_names.json.
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;

import '../models/classification_result.dart';
import 'image_classifier_service.dart';
import 'image_preprocessing.dart';
import 'onnx_service.dart';

class DermatologyInferenceService implements ImageClassifierService {
  static const _modelAsset = 'assets/models/dermnet_model.onnx';
  static const _classNamesAsset = 'assets/models/dermnet_class_names.json';

  // This model is an EfficientNetV2-S export whose input rescaling/
  // normalization is baked into the ONNX graph itself (confirmed: the graph
  // has Sub/Mul/Cast ops right after the input, i.e. it does its own
  // preprocessing). It expects RAW [0, 255] pixel values — do NOT switch
  // this to PreprocessMode.imagenet (or any other normalized mode); that
  // would double-preprocess the input and silently produce wrong
  // predictions with no error thrown. See image_preprocessing.dart and
  // README.md ("Swapping in updated models") for the full explanation.
  static const _preprocessMode = PreprocessMode.raw255;

  final OnnxService _onnx = OnnxService(assetPath: _modelAsset);
  List<String>? _classNames;

  @override
  String get displayName => 'Dermatology';

  @override
  bool get isLoaded => _onnx.isLoaded && _classNames != null;

  @override
  Future<void> load() async {
    await _onnx.load();
    final raw = await rootBundle.loadString(_classNamesAsset);
    final names = (jsonDecode(raw) as List).cast<String>();
    if (names.length != 11) {
      throw OnnxLoadException(
        '$_classNamesAsset has ${names.length} classes, expected 11. '
        'Check the file matches the model this app was told to bundle.',
      );
    }
    _classNames = names;
  }

  @override
  Future<List<ClassificationResult>> classify(File imageFile) async {
    final classNames = _classNames;
    if (!isLoaded || classNames == null) {
      throw StateError('DermatologyInferenceService.load() must complete before classify().');
    }

    final bytes = await imageFile.readAsBytes();
    final tensor = preprocessImageForModel(
      imageBytes: bytes,
      spec: _onnx.inputSpec,
      mode: _preprocessMode,
    );
    final raw = await _onnx.run(tensor.data, tensor.shape);
    return classificationResultsFromLogits(raw, classNames);
  }

  @override
  Future<void> dispose() => _onnx.dispose();
}
