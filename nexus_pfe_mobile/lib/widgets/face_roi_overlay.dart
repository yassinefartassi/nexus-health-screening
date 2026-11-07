// Draws the detected face's bounding box (and the forehead ROI band read
// for CHROM/POS) on top of the live camera preview.
//
// The tricky part: CameraImage buffers arrive in the sensor's native
// (landscape) pixel orientation, while CameraPreview displays an
// already-rotated-to-upright texture sized to a portrait AspectRatio box.
// To align the overlay with what's on screen, every point is rotated the
// same way ML Kit was told the buffer is rotated (see
// rppg_screen.dart._rotationForCamera), then scaled uniformly into the
// widget's box — no letterboxing/cropping is needed because that portrait
// box's aspect ratio is derived from the same previewSize, so it matches
// the rotated buffer's aspect ratio exactly.
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceRoiOverlay extends CustomPainter {
  final Rect? faceBox;
  final Size imageSize;
  final InputImageRotation rotation;

  FaceRoiOverlay({
    required this.faceBox,
    required this.imageSize,
    required this.rotation,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final box = faceBox;
    if (box == null || imageSize.isEmpty) return;

    final uprightSize = _uprightSize(imageSize, rotation);
    if (uprightSize.width == 0 || uprightSize.height == 0) return;
    final scale = size.width / uprightSize.width;

    final topLeft = _rotatePoint(Offset(box.left, box.top), imageSize, rotation) * scale;
    final bottomRight =
        _rotatePoint(Offset(box.right, box.bottom), imageSize, rotation) * scale;
    final faceRect = Rect.fromPoints(topLeft, bottomRight);

    final facePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.greenAccent;
    canvas.drawRRect(
      RRect.fromRectAndRadius(faceRect, const Radius.circular(8)),
      facePaint,
    );

    // Same forehead-band geometry as RppgService._foreheadRoi, drawn purely
    // for visual feedback so the user can see what's actually being sampled.
    final fx1 = box.left + 0.30 * box.width;
    final fx2 = box.left + 0.70 * box.width;
    final fy1 = box.top + 0.08 * box.height;
    final fy2 = box.top + 0.25 * box.height;
    final roiTopLeft = _rotatePoint(Offset(fx1, fy1), imageSize, rotation) * scale;
    final roiBottomRight = _rotatePoint(Offset(fx2, fy2), imageSize, rotation) * scale;
    final roiRect = Rect.fromPoints(roiTopLeft, roiBottomRight);

    final roiPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.amberAccent.withValues(alpha: 0.45);
    canvas.drawRect(roiRect, roiPaint);
  }

  static Offset _rotatePoint(Offset p, Size bufferSize, InputImageRotation rotation) {
    switch (rotation) {
      case InputImageRotation.rotation90deg:
        return Offset(bufferSize.height - p.dy, p.dx);
      case InputImageRotation.rotation180deg:
        return Offset(bufferSize.width - p.dx, bufferSize.height - p.dy);
      case InputImageRotation.rotation270deg:
        return Offset(p.dy, bufferSize.width - p.dx);
      case InputImageRotation.rotation0deg:
        return p;
    }
  }

  static Size _uprightSize(Size bufferSize, InputImageRotation rotation) {
    final swapped = rotation == InputImageRotation.rotation90deg ||
        rotation == InputImageRotation.rotation270deg;
    return swapped ? Size(bufferSize.height, bufferSize.width) : bufferSize;
  }

  @override
  bool shouldRepaint(covariant FaceRoiOverlay oldDelegate) {
    return oldDelegate.faceBox != faceBox ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.rotation != rotation;
  }
}
