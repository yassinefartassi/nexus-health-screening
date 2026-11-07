// Live rPPG screen: camera preview + face ROI overlay + running BPM/HRV.
//
// Camera lifecycle (CameraController, permission, image stream, device
// rotation) lives here; all vision/signal-processing logic is delegated to
// RppgService + rppg_algorithms.dart.
import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../models/rppg_result.dart';
import '../services/permission_service.dart';
import '../services/rppg_service.dart';
import '../widgets/face_roi_overlay.dart';
import '../widgets/waveform_sparkline.dart';

class RppgScreen extends StatefulWidget {
  const RppgScreen({super.key});

  @override
  State<RppgScreen> createState() => _RppgScreenState();
}

enum _Stage { checkingPermission, permissionDenied, initializingCamera, cameraError, running }

class _RppgScreenState extends State<RppgScreen> with WidgetsBindingObserver {
  final RppgService _rppgService = RppgService();

  _Stage _stage = _Stage.checkingPermission;
  String? _cameraError;

  CameraController? _controller;
  CameraDescription? _camera;
  InputImageRotation _lastRotation = InputImageRotation.rotation0deg;

  Timer? _resultTimer;
  RppgResult? _result;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _start();
  }

  Future<void> _start() async {
    final granted = await PermissionService.ensureCameraPermission();
    if (!mounted) return;
    if (!granted) {
      setState(() => _stage = _Stage.permissionDenied);
      return;
    }
    await _initCamera();
  }

  Future<void> _initCamera() async {
    setState(() => _stage = _Stage.initializingCamera);
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw CameraException('noCamera', 'No camera available on this device.');
      }
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        // A single densely-packed plane per frame keeps RppgService's pixel
        // sampling simple (see rppg_service.dart's header comment).
        imageFormatGroup:
            Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
      );
      await controller.initialize();
      await controller.startImageStream(_onFrame);

      if (!mounted) return;
      setState(() {
        _controller = controller;
        _camera = camera;
        _stage = _Stage.running;
      });

      _resultTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        if (!mounted) return;
        setState(() => _result = _rppgService.computeResult());
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _stage = _Stage.cameraError;
        _cameraError = e.toString();
      });
    }
  }

  void _onFrame(CameraImage image) {
    final camera = _camera;
    final controller = _controller;
    if (camera == null || controller == null) return;

    final rotation = _rotationForCamera(camera, controller.value.deviceOrientation);
    if (rotation == null) return;
    _lastRotation = rotation;

    // Fire-and-forget: RppgService throttles the expensive face-detection
    // call internally and must not block this stream callback.
    unawaited(_rppgService.processFrame(image: image, rotation: rotation));
  }

  /// Standard ML Kit rotation-compensation formula (see google_mlkit_commons
  /// example app) so face detection stays correct across device rotation
  /// for both front and back cameras.
  InputImageRotation? _rotationForCamera(
    CameraDescription camera,
    DeviceOrientation deviceOrientation,
  ) {
    if (Platform.isIOS) {
      return InputImageRotationValue.fromRawValue(camera.sensorOrientation);
    }
    if (!Platform.isAndroid) return null;

    const orientations = {
      DeviceOrientation.portraitUp: 0,
      DeviceOrientation.landscapeLeft: 90,
      DeviceOrientation.portraitDown: 180,
      DeviceOrientation.landscapeRight: 270,
    };
    final base = orientations[deviceOrientation];
    if (base == null) return null;

    final rotationCompensation = camera.lensDirection == CameraLensDirection.front
        ? (camera.sensorOrientation + base) % 360
        : (camera.sensorOrientation - base + 360) % 360;
    return InputImageRotationValue.fromRawValue(rotationCompensation);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      controller.stopImageStream();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _resultTimer?.cancel();
    _controller?.dispose();
    _rppgService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('rPPG')),
      body: SafeArea(child: _buildBody()),
    );
  }

  Widget _buildBody() {
    switch (_stage) {
      case _Stage.checkingPermission:
      case _Stage.initializingCamera:
        return const Center(child: CircularProgressIndicator());
      case _Stage.permissionDenied:
        return _ErrorView(
          icon: Icons.videocam_off_outlined,
          title: 'Camera permission needed',
          message: 'rPPG needs the camera to measure your pulse from your face. '
              'Grant access to continue.',
          actionLabel: 'Grant permission',
          onAction: _start,
          secondaryActionLabel: 'Open settings',
          onSecondaryAction: PermissionService.openSettings,
        );
      case _Stage.cameraError:
        return _ErrorView(
          icon: Icons.error_outline,
          title: 'Camera error',
          message: _cameraError ?? 'Unknown error',
          actionLabel: 'Retry',
          onAction: _initCamera,
        );
      case _Stage.running:
        return _buildRunningBody();
    }
  }

  Widget _buildRunningBody() {
    final controller = _controller!;
    final result = _result;
    final faceFound = _rppgService.faceFound;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: AspectRatio(
            aspectRatio: 1 / controller.value.aspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),
                CustomPaint(
                  painter: FaceRoiOverlay(
                    faceBox: _rppgService.lastFaceBox,
                    imageSize: _rppgService.lastImageSize ?? Size.zero,
                    rotation: _lastRotation,
                  ),
                ),
                Positioned(
                  top: 10,
                  left: 10,
                  child: _StatusChip(faceFound: faceFound),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: _rppgService.bufferFillFraction,
            minHeight: 6,
            backgroundColor: Colors.black.withValues(alpha: 0.06),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          faceFound
              ? 'Hold still, face the camera, ensure even lighting. '
                  'First reading takes ~10 seconds.'
              : 'No face detected — center your face in frame.',
          style: TextStyle(color: faceFound ? Colors.black54 : Colors.red),
        ),
        const SizedBox(height: 20),
        if (result == null)
          const Center(
            child: Padding(
              padding: EdgeInsets.symmetric(vertical: 24),
              child: Text('Calibrating…', style: TextStyle(color: Colors.black54)),
            ),
          )
        else ...[
          Row(
            children: [
              Expanded(
                child: _MetricCard(
                  label: 'CHROM',
                  value: result.bpmChrom > 0 ? result.bpmChrom.toStringAsFixed(0) : '--',
                  unit: 'BPM',
                  color: Colors.deepOrange,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _MetricCard(
                  label: 'POS',
                  value: result.bpmPos > 0 ? result.bpmPos.toStringAsFixed(0) : '--',
                  unit: 'BPM',
                  color: Colors.blue,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _MetricCard(
                  label: 'SDNN',
                  value: result.hrv.beatsDetected >= 3
                      ? result.hrv.sdnnMs.toStringAsFixed(0)
                      : '--',
                  unit: 'ms',
                  color: Colors.purple,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _MetricCard(
                  label: 'RMSSD',
                  value: result.hrv.beatsDetected >= 3
                      ? result.hrv.rmssdMs.toStringAsFixed(0)
                      : '--',
                  unit: 'ms',
                  color: Colors.pink,
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Text('Waveform comparison', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          WaveformSparkline(
            samples: result.chromWaveform,
            color: Colors.deepOrange,
            label: 'CHROM',
          ),
          const SizedBox(height: 12),
          WaveformSparkline(
            samples: result.posWaveform,
            color: Colors.blue,
            label: 'POS',
          ),
        ],
        const SizedBox(height: 20),
        const Text(
          '⚠️ Estimate only, not a certified medical device. Do not use for '
          'clinical decisions.',
          style: TextStyle(color: Colors.black45, fontSize: 12),
        ),
      ],
    );
  }
}

class _StatusChip extends StatelessWidget {
  final bool faceFound;
  const _StatusChip({required this.faceFound});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.circle,
            size: 10,
            color: faceFound ? Colors.greenAccent : Colors.redAccent,
          ),
          const SizedBox(width: 6),
          Text(
            faceFound ? 'Face detected' : 'No face',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final String unit;
  final Color color;

  const _MetricCard({
    required this.label,
    required this.value,
    required this.unit,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: color.withValues(alpha: 0.08),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Column(
          children: [
            Text(label, style: TextStyle(color: color, fontWeight: FontWeight.w600)),
            const SizedBox(height: 4),
            Text(
              value,
              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            Text(unit, style: const TextStyle(color: Colors.black45, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  final IconData icon;
  final String title;
  final String message;
  final String actionLabel;
  final VoidCallback onAction;
  final String? secondaryActionLabel;
  final VoidCallback? onSecondaryAction;

  const _ErrorView({
    required this.icon,
    required this.title,
    required this.message,
    required this.actionLabel,
    required this.onAction,
    this.secondaryActionLabel,
    this.onSecondaryAction,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 48, color: Colors.redAccent),
            const SizedBox(height: 12),
            Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton(onPressed: onAction, child: Text(actionLabel)),
            if (secondaryActionLabel != null && onSecondaryAction != null) ...[
              const SizedBox(height: 8),
              TextButton(onPressed: onSecondaryAction, child: Text(secondaryActionLabel!)),
            ],
          ],
        ),
      ),
    );
  }
}
