import 'dart:io';

import '../models/classification_result.dart';

/// Common contract implemented by [OcularInferenceService] and
/// [DermatologyInferenceService] so `lib/widgets/image_classification_screen.dart`
/// can drive either one without knowing which model it's talking to.
abstract class ImageClassifierService {
  /// Human-readable module name shown in the UI (e.g. "Ocular").
  String get displayName;

  bool get isLoaded;

  /// Loads the ONNX model + class names from assets. Throws
  /// [OnnxLoadException] on failure — callers should show an error state
  /// rather than let the user attempt inference against a half-loaded model.
  Future<void> load();

  /// Runs preprocessing + inference + label mapping on [imageFile], returning
  /// results sorted by descending confidence.
  Future<List<ClassificationResult>> classify(File imageFile);

  Future<void> dispose();
}
