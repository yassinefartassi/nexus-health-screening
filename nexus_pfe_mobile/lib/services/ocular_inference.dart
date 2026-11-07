// Ocular (eye disease) classifier: Cataract / Conjunctivitis / Eyelid /
// Normal / Uveitis, 5 classes. Model and class order match the desktop
// prototype's `models/eye_disease_model.onnx` +
// `models/class_names/oculaire_classes.json` 1:1 — do not reorder the JSON
// without re-checking against the Kaggle training notebook's class_indices.
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;

import '../models/classification_result.dart';
import 'image_classifier_service.dart';
import 'image_preprocessing.dart';
import 'onnx_service.dart';

class OcularInferenceService implements ImageClassifierService {
  static const _modelAsset = 'assets/models/eye_disease_model_merged.onnx';
  static const _classNamesAsset = 'assets/models/eye_disease_class_names.json';

  // Matches OCULAIRE_PREPROCESS = "imagenet" in the desktop prototype's config.py.
  static const _preprocessMode = PreprocessMode.imagenet;

  final OnnxService _onnx = OnnxService(assetPath: _modelAsset);
  List<String>? _classNames;

  @override
  String get displayName => 'Ocular';

  @override
  bool get isLoaded => _onnx.isLoaded && _classNames != null;

  @override
  Future<void> load() async {
    await _onnx.load();
    final raw = await rootBundle.loadString(_classNamesAsset);
    final names = (jsonDecode(raw) as List).cast<String>();
    if (names.length != 5) {
      throw OnnxLoadException(
        '$_classNamesAsset has ${names.length} classes, expected 5. '
        'Check the file matches the model this app was told to bundle.',
      );
    }
    _classNames = names;
  }

  @override
  Future<List<ClassificationResult>> classify(File imageFile) async {
    final classNames = _classNames;
    if (!isLoaded || classNames == null) {
      throw StateError('OcularInferenceService.load() must complete before classify().');
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
