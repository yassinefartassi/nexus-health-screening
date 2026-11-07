// On-device smoke test for both bundled ONNX models.
//
// This exists specifically to catch, at test time, the two failure modes
// that previously only surfaced at runtime on a real device:
//   1. A broken/incomplete asset (e.g. the ocular model's merge into a
//      single self-contained .onnx file silently failing, leaving external
//      weight references that can't resolve inside the Flutter asset
//      bundle) -> service.load() throws.
//   2. Wrong per-model preprocessing (e.g. accidentally re-unifying the
//      ocular and dermatology preprocessing modes) -> classify() throws
//      (shape mismatch) or returns a degenerate/empty result.
//
// This is a real ONNX Runtime load + inference, so it needs the actual
// platform plugin (Android/iOS native bindings) and cannot run under plain
// `flutter test` (which executes on a headless host VM with no platform
// channels). Run it against a connected device or emulator:
//
//   flutter test integration_test/model_smoke_test.dart
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:integration_test/integration_test.dart';

import 'package:nexus_pfe_mobile/services/dermatology_inference.dart';
import 'package:nexus_pfe_mobile/services/ocular_inference.dart';

/// Writes an arbitrary (content doesn't matter — only that both models
/// accept it, run inference, and return a well-formed result) dummy image
/// to a temp file and returns it.
Future<File> _writeDummyImage() async {
  final image = img.Image(width: 256, height: 256);
  img.fill(image, color: img.ColorRgb8(120, 90, 60));
  final file = File(
    '${Directory.systemTemp.path}/nexus_pfe_dummy_input.png',
  );
  await file.writeAsBytes(img.encodePng(image));
  return file;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Ocular model loads from assets and runs a dummy inference', (
    tester,
  ) async {
    final service = OcularInferenceService();
    addTearDown(service.dispose);

    await service.load();
    expect(service.isLoaded, isTrue);

    final dummy = await _writeDummyImage();
    final results = await service.classify(dummy);

    expect(results, hasLength(5), reason: 'Ocular model has 5 output classes');
    for (final r in results) {
      expect(r.confidence, inInclusiveRange(0.0, 1.0));
      expect(r.label, isNotEmpty);
    }
  });

  testWidgets(
    'Dermatology model loads from assets and runs a dummy inference',
    (tester) async {
      final service = DermatologyInferenceService();
      addTearDown(service.dispose);

      await service.load();
      expect(service.isLoaded, isTrue);

      final dummy = await _writeDummyImage();
      final results = await service.classify(dummy);

      expect(
        results,
        hasLength(11),
        reason: 'Dermatology model has 11 output classes',
      );
      for (final r in results) {
        expect(r.confidence, inInclusiveRange(0.0, 1.0));
        expect(r.label, isNotEmpty);
      }
    },
  );
}
