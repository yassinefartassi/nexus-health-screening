import 'package:flutter/material.dart';

/// Minimal line-chart of a filtered pulse waveform, used to let CHROM and
/// POS be compared visually side by side (per the thesis's evaluation goal).
class WaveformSparkline extends StatelessWidget {
  final List<double> samples;
  final Color color;
  final String label;

  const WaveformSparkline({
    super.key,
    required this.samples,
    required this.color,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: Theme.of(context).textTheme.labelMedium),
        const SizedBox(height: 4),
        SizedBox(
          height: 60,
          width: double.infinity,
          child: CustomPaint(
            painter: _SparklinePainter(samples: samples, color: color),
          ),
        ),
      ],
    );
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> samples;
  final Color color;

  _SparklinePainter({required this.samples, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final backgroundPaint = Paint()..color = Colors.black.withValues(alpha: 0.04);
    canvas.drawRRect(
      RRect.fromRectAndRadius(Offset.zero & size, const Radius.circular(8)),
      backgroundPaint,
    );

    if (samples.length < 2) return;

    final minV = samples.reduce((a, b) => a < b ? a : b);
    final maxV = samples.reduce((a, b) => a > b ? a : b);
    final range = (maxV - minV).abs() < 1e-9 ? 1.0 : (maxV - minV);

    final path = Path();
    for (int i = 0; i < samples.length; i++) {
      final x = size.width * i / (samples.length - 1);
      final normalized = (samples[i] - minV) / range;
      final y = size.height * (1 - normalized);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = color;
    canvas.drawPath(path, linePaint);
  }

  @override
  bool shouldRepaint(covariant _SparklinePainter oldDelegate) {
    return oldDelegate.samples != samples || oldDelegate.color != color;
  }
}
