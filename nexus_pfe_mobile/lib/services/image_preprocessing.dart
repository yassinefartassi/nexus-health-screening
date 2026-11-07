// Image preprocessing shared by every on-device image classifier.
//
// These steps — resize, then per-[PreprocessMode] scaling/normalization —
// MUST exactly match what each model was trained/exported with, or
// predictions will be silently wrong (the model will still run, it will
// just be scoring pixels that don't look like what it was trained on).
// This mirrors `common_onnx.py._preprocess()` from the desktop prototype
// (NEXUS_PFE).
//
// IMPORTANT — preprocessing is per-model, NOT shared across models:
//   - Ocular (`eye_disease_model_merged.onnx`, ResNet-style): trained with
//     external ImageNet normalization -> use [PreprocessMode.imagenet].
//   - Dermatology (`dermnet_model.onnx`, EfficientNetV2-S): normalization is
//     baked into the model graph itself (its first layers do their own
//     rescaling), so the Dart side must hand it RAW [0, 255] pixel values
//     -> use [PreprocessMode.raw255]. Applying ImageNet (or any other)
//     normalization on top double-preprocesses the input: it doesn't crash,
//     it just silently produces wrong predictions.
// Do NOT unify these two modes "for consistency" — they are different on
// purpose because the two models were exported with different preprocessing
// contracts. See `ocular_inference.dart` / `dermatology_inference.dart` for
// which mode each classifier passes.
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'onnx_service.dart';

/// Normalization scheme applied to decoded pixel values before they're fed
/// to the model. Which mode is correct is a property of how the specific
/// model was trained/exported — see the file-level comment above.
///
/// - [imagenet]: scale to [0, 1], then subtract ImageNet mean / divide by
///   ImageNet std. Standard for fine-tuned ResNet/MobileNet-style backbones
///   that do NOT bake normalization into the graph. Used by the ocular
///   model.
/// - [zeroToOne]: scale to [0, 1], no further normalization.
/// - [minusOneToOne]: scale to [0, 1], then rescale to [-1, 1]
///   (i.e. `(x - 0.5) / 0.5`).
/// - [raw255]: leave pixels as raw `0-255` floats — no scaling, no
///   normalization at all. Required for models (e.g. Keras
///   EfficientNetV2-S exports, like the dermatology model) whose graph
///   already contains its own input-rescaling/normalization layers.
enum PreprocessMode { imagenet, zeroToOne, minusOneToOne, raw255 }

/// mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] in R, G, B order —
/// the standard ImageNet statistics used by the training notebooks.
const List<double> imagenetMean = [0.485, 0.456, 0.406];
const List<double> imagenetStd = [0.229, 0.224, 0.225];

/// A tensor ready to feed to [OnnxService.run]: flattened data + its shape.
class PreprocessedTensor {
  final Float32List data;
  final List<int> shape;
  const PreprocessedTensor(this.data, this.shape);
}

/// Decodes [imageBytes], resizes to the model's expected (width x height)
/// per [spec], scales to [0, 1], applies [mode] normalization, and lays the
/// result out as NCHW or NHWC per [spec.channelsFirst].
PreprocessedTensor preprocessImageForModel({
  required Uint8List imageBytes,
  required OnnxInputSpec spec,
  required PreprocessMode mode,
}) {
  final decoded = img.decodeImage(imageBytes);
  if (decoded == null) {
    throw const FormatException('Could not decode image bytes.');
  }

  // Bilinear resize to the model's exact expected input size.
  final resized = img.copyResize(
    decoded,
    width: spec.width,
    height: spec.height,
    interpolation: img.Interpolation.linear,
  );

  final height = spec.height;
  final width = spec.width;
  final data = Float32List(1 * spec.channels * height * width);

  int nchwIndex(int c, int y, int x) => (c * height + y) * width + x;
  int nhwcIndex(int y, int x, int c) => (y * width + x) * spec.channels + c;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final pixel = resized.getPixel(x, y);
      // HWC, raw [0, 255] — scaling (if any) is applied per-mode below, not
      // here, because [PreprocessMode.raw255] must NOT be scaled at all.
      final rgb255 = [pixel.r.toDouble(), pixel.g.toDouble(), pixel.b.toDouble()];

      for (int c = 0; c < 3 && c < spec.channels; c++) {
        double value;
        switch (mode) {
          case PreprocessMode.raw255:
            // Model's own graph does its own rescaling/normalization — do
            // not touch the pixel value here.
            value = rgb255[c];
            break;
          case PreprocessMode.imagenet:
            value = (rgb255[c] / 255.0 - imagenetMean[c]) / imagenetStd[c];
            break;
          case PreprocessMode.minusOneToOne:
            value = (rgb255[c] / 255.0 - 0.5) / 0.5;
            break;
          case PreprocessMode.zeroToOne:
            value = rgb255[c] / 255.0;
            break;
        }

        final index = spec.channelsFirst
            ? nchwIndex(c, y, x)
            : nhwcIndex(y, x, c);
        data[index] = value;
      }
    }
  }

  return PreprocessedTensor(data, spec.tensorShape);
}
