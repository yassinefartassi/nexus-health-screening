import 'dart:math' show exp;

/// A single labeled prediction with its confidence score in [0, 1].
class ClassificationResult {
  final String label;
  final double confidence;

  const ClassificationResult({required this.label, required this.confidence});

  @override
  String toString() =>
      '$label (${(confidence * 100).toStringAsFixed(1)}%)';
}

/// Converts a model's raw output vector into a confidence-sorted list of
/// [ClassificationResult]s.
///
/// Mirrors the desktop prototype's logic in `common_onnx.py`: if the raw
/// values already look like a probability distribution (sum ~1, all >= 0)
/// they're used as-is; otherwise they're treated as logits and passed
/// through softmax. This matters because some ONNX exports bake softmax
/// into the graph and some don't — applying softmax twice would silently
/// flatten every confidence score toward uniform.
List<ClassificationResult> classificationResultsFromLogits(
  List<double> raw,
  List<String> classNames,
) {
  final n = raw.length < classNames.length ? raw.length : classNames.length;
  final scores = raw.sublist(0, n);

  final sum = scores.fold<double>(0, (a, b) => a + b);
  final looksLikeProbabilities =
      sum >= 0.99 && sum <= 1.01 && scores.every((v) => v >= 0);

  final probs = looksLikeProbabilities ? scores : _softmax(scores);

  final results = List<ClassificationResult>.generate(
    n,
    (i) => ClassificationResult(label: classNames[i], confidence: probs[i]),
  );
  results.sort((a, b) => b.confidence.compareTo(a.confidence));
  return results;
}

List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce((a, b) => a > b ? a : b);
  final exps = logits.map((v) => exp(v - maxLogit)).toList();
  final sumExps = exps.fold<double>(0, (a, b) => a + b);
  return exps.map((v) => v / sumExps).toList();
}
