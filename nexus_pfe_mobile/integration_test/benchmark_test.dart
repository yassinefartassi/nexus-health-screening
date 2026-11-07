// On-device benchmark for both bundled ONNX models -- the Flutter half of
// the cross-platform benchmark suite (see benchmark/SCHEMA.md at the repo
// root; benchmark/webapp-benchmark/bench.py is the reference implementation
// this mirrors: same N=30 timed runs / 5 warmup runs discarded, same
// mean/median/p95 math, same per-image accuracy scoring against
// benchmark/labels.json).
//
// Like model_smoke_test.dart, this needs the real platform ONNX Runtime
// plugin and cannot run under plain `flutter test` (headless host VM, no
// platform channels). Run it against a connected device or emulator:
//
//   flutter test integration_test/benchmark_test.dart
//
// Integration tests can't return a value to the host process, so the result
// (one JSON object per SCHEMA.md's top-level shape) is printed as a SINGLE
// LINE prefixed with `BENCHMARK_JSON:` -- see ../../benchmark/flutter-benchmark/run_benchmark.ps1,
// which greps stdout for that marker and writes results/flutter_results.json.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:nexus_pfe_mobile/services/dermatology_inference.dart';
import 'package:nexus_pfe_mobile/services/image_classifier_service.dart';
import 'package:nexus_pfe_mobile/services/ocular_inference.dart';

const int kRuns = 30;
const int kWarmup = 5;

/// mean/median/p95 (ms) over timed calls, warmup discarded -- same math as
/// benchmark/bench_core.py::time_calls, so the two are directly comparable.
class LatencyStats {
  final double mean;
  final double median;
  final double p95;
  final int n;
  final int warmup;

  LatencyStats({
    required this.mean,
    required this.median,
    required this.p95,
    required this.n,
    required this.warmup,
  });

  Map<String, dynamic> toJson() => {
    'mean': mean,
    'median': median,
    'p95': p95,
    'n': n,
    'warmup': warmup,
  };
}

Future<LatencyStats> timeCalls(
  Future<void> Function() fn, {
  int runs = kRuns,
  int warmup = kWarmup,
}) async {
  for (var i = 0; i < warmup; i++) {
    await fn();
  }

  final samplesMs = <double>[];
  for (var i = 0; i < runs; i++) {
    final sw = Stopwatch()..start();
    await fn();
    sw.stop();
    samplesMs.add(sw.elapsedMicroseconds / 1000.0);
  }

  samplesMs.sort();
  final p95Index = math.min(
    (0.95 * (samplesMs.length - 1)).round(),
    samplesMs.length - 1,
  );
  final mean = samplesMs.reduce((a, b) => a + b) / samplesMs.length;
  final mid = samplesMs.length ~/ 2;
  final median = samplesMs.length.isOdd
      ? samplesMs[mid]
      : (samplesMs[mid - 1] + samplesMs[mid]) / 2.0;

  return LatencyStats(
    mean: mean,
    median: median,
    p95: samplesMs[p95Index],
    n: runs,
    warmup: warmup,
  );
}

/// Best-effort process RSS in MB via dart:io's ProcessInfo. Returns null
/// (never throws) if unavailable on this platform -- matches
/// bench_core.py::get_memory_mb's "-1 rather than failing the whole
/// benchmark" philosophy, just represented as null in JSON here.
double? currentMemoryMb() {
  try {
    return ProcessInfo.currentRss / (1024 * 1024);
  } catch (_) {
    return null;
  }
}

Future<Map<String, dynamic>> benchmarkModel({
  required String modelKey,
  required String modelPath,
  required int numClasses,
  required ImageClassifierService Function() makeService,
  required List<File> testImages,
  required Map<String, dynamic> labelsForModel,
}) async {
  final service = makeService();

  final loadSw = Stopwatch()..start();
  await service.load();
  loadSw.stop();
  expect(service.isLoaded, isTrue, reason: '$modelKey model failed to load');

  final benchImage = testImages.first;
  final latency = await timeCalls(() async {
    await service.classify(benchImage);
  });

  final memoryMb = currentMemoryMb();

  final predictions = <Map<String, dynamic>>[];
  var correct = 0;
  var scored = 0;
  final scoredImages = <String>[];

  for (final imageFile in testImages) {
    final fileName = imageFile.uri.pathSegments.last;
    final results = await service.classify(imageFile);
    final top = results.first;
    final expected = labelsForModel[fileName] as String?;

    predictions.add({
      'image': fileName,
      'predicted_class': top.label,
      'confidence': top.confidence,
      'expected_class': expected,
    });

    if (expected != null) {
      scored++;
      scoredImages.add(fileName);
      if (top.label == expected) correct++;
    }
  }

  await service.dispose();

  return {
    'model_path': modelPath,
    'num_classes': numClasses,
    'load_time_ms': loadSw.elapsedMicroseconds / 1000.0,
    'latency_ms': latency.toJson(),
    'memory_mb': memoryMb,
    'accuracy': {
      'correct': correct,
      'total': scored,
      'accuracy': scored > 0 ? correct / scored : null,
      'scored_images': scoredImages,
    },
    'predictions': predictions,
  };
}

/// Copies a bundled asset image to a real on-disk temp file -- classify()
/// takes a File, and asset bytes aren't directly usable as one.
Future<File> _assetToTempFile(String assetPath, String tempName) async {
  final bytes = await rootBundle.load(assetPath);
  final file = File('${Directory.systemTemp.path}/$tempName');
  await file.writeAsBytes(bytes.buffer.asUint8List());
  return file;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Benchmark both models and emit BENCHMARK_JSON', (
    tester,
  ) async {
    final labelsRaw = await rootBundle.loadString(
      'assets/test_images/labels.json',
    );
    final labels = jsonDecode(labelsRaw) as Map<String, dynamic>;

    const dermatoNames = [
      '12174-rosacea.jpg',
      'Atopic Dermatitis.jpg',
      'Molluscum-contagiosum_1.jpg',
    ];
    const ocularNames = ['Eyeliddisorders.jpg', 'Uveitis.jpg', 'images.jpg'];

    final dermatoImages = <File>[];
    for (final name in dermatoNames) {
      dermatoImages.add(
        await _assetToTempFile('assets/test_images/dermatho/$name', name),
      );
    }
    final ocularImages = <File>[];
    for (final name in ocularNames) {
      ocularImages.add(
        await _assetToTempFile('assets/test_images/occulaire/$name', name),
      );
    }

    final dermatologyResult = await benchmarkModel(
      modelKey: 'dermatology',
      modelPath: 'assets/models/dermnet_model.onnx',
      numClasses: 11,
      makeService: () => DermatologyInferenceService(),
      testImages: dermatoImages,
      labelsForModel: labels['dermatology'] as Map<String, dynamic>,
    );

    final ocularResult = await benchmarkModel(
      modelKey: 'ocular',
      modelPath: 'assets/models/eye_disease_model_merged.onnx',
      numClasses: 5,
      makeService: () => OcularInferenceService(),
      testImages: ocularImages,
      labelsForModel: labels['ocular'] as Map<String, dynamic>,
    );

    // No headless rPPG entrypoint exists in this app's services layer (it's
    // camera-driven, UI-integrated) -- omit rather than fake a number. See
    // benchmark/flutter-benchmark/README.md for the full explanation.
    final results = {
      'platform': 'flutter',
      'device': '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
      'models': {'dermatology': dermatologyResult, 'ocular': ocularResult},
      'rppg': null,
    };

    // Single line, compact JSON, unique marker prefix -- see run_benchmark.ps1.
    // ignore: avoid_print
    print('BENCHMARK_JSON:${jsonEncode(results)}');

    expect(dermatologyResult['predictions'], hasLength(3));
    expect(ocularResult['predictions'], hasLength(3));
  });
}
