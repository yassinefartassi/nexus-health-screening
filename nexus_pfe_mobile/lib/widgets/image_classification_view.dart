// Shared UI + control flow for the two static-image classifiers (Ocular,
// Dermatology). The two screens differ only in which
// [ImageClassifierService] they hand this widget — all picking, loading,
// error, and results-display logic lives here once.
import 'dart:developer' as developer;
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/classification_result.dart';
import '../services/image_classifier_service.dart';
import '../services/permission_service.dart';

class ImageClassificationView extends StatefulWidget {
  final ImageClassifierService service;
  final Color accentColor;

  const ImageClassificationView({
    super.key,
    required this.service,
    required this.accentColor,
  });

  @override
  State<ImageClassificationView> createState() => _ImageClassificationViewState();
}

enum _LoadState { loading, ready, failed }

class _ImageClassificationViewState extends State<ImageClassificationView> {
  _LoadState _loadState = _LoadState.loading;
  String? _loadError;

  File? _pickedImage;
  List<ClassificationResult>? _results;
  bool _classifying = false;
  String? _classifyError;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await widget.service.load();
      if (!mounted) return;
      setState(() => _loadState = _LoadState.ready);
    } catch (e, st) {
      // Always use toString()/full detail here, not OnnxLoadException.message
      // alone — .message is just the generic wrapper text ("Failed to load
      // ONNX model at ..."); the actual root cause lives in .cause and was
      // previously being silently dropped, making this failure mode
      // undiagnosable from a device report.
      developer.log(
        'Model load failed for ${widget.service.displayName}',
        name: 'ImageClassificationView',
        error: e,
        stackTrace: st,
      );
      if (!mounted) return;
      setState(() {
        _loadState = _LoadState.failed;
        _loadError = e.toString();
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    if (source == ImageSource.camera) {
      final granted = await PermissionService.ensureCameraPermission();
      if (!granted) {
        if (!mounted) return;
        _showPermissionDeniedDialog();
        return;
      }
    }

    final picker = ImagePicker();
    final XFile? picked = await picker.pickImage(
      source: source,
      maxWidth: 2048,
      imageQuality: 95,
    );
    if (picked == null) return;

    setState(() {
      _pickedImage = File(picked.path);
      _results = null;
      _classifyError = null;
    });
    await _runClassification();
  }

  Future<void> _runClassification() async {
    final image = _pickedImage;
    if (image == null) return;

    setState(() {
      _classifying = true;
      _classifyError = null;
    });
    try {
      final results = await widget.service.classify(image);
      if (!mounted) return;
      setState(() {
        _results = results;
        _classifying = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _classifyError = e.toString();
        _classifying = false;
      });
    }
  }

  void _showPermissionDeniedDialog() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Camera permission needed'),
        content: const Text(
          'This app needs camera access to take a photo. You can grant it '
          'in system settings.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              PermissionService.openSettings();
            },
            child: const Text('Open settings'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.service.displayName)),
      body: switch (_loadState) {
        _LoadState.loading => const Center(child: CircularProgressIndicator()),
        _LoadState.failed => _ModelLoadErrorView(
            message: _loadError ?? 'Unknown error',
            onRetry: () {
              setState(() => _loadState = _LoadState.loading);
              _loadModel();
            },
          ),
        _LoadState.ready => _buildReadyBody(),
      },
    );
  }

  Widget _buildReadyBody() {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (_pickedImage != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(_pickedImage!, height: 260, fit: BoxFit.cover),
            )
          else
            Container(
              height: 260,
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.04),
                borderRadius: BorderRadius.circular(12),
              ),
              alignment: Alignment.center,
              child: const Text(
                'No image selected',
                style: TextStyle(color: Colors.black45),
              ),
            ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt_outlined),
                  label: const Text('Camera'),
                  style: FilledButton.styleFrom(backgroundColor: widget.accentColor),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Gallery'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          if (_classifying) const Center(child: CircularProgressIndicator()),
          if (_classifyError != null)
            Text(
              'Inference failed: $_classifyError',
              style: const TextStyle(color: Colors.red),
            ),
          if (_results != null) _ResultsList(results: _results!, accentColor: widget.accentColor),
        ],
      ),
    );
  }
}

class _ResultsList extends StatelessWidget {
  final List<ClassificationResult> results;
  final Color accentColor;

  const _ResultsList({required this.results, required this.accentColor});

  @override
  Widget build(BuildContext context) {
    if (results.isEmpty) {
      return const Text('No prediction returned.');
    }
    final top = results.first;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Prediction', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Card(
          color: accentColor.withValues(alpha: 0.08),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    top.label,
                    style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                  ),
                ),
                Text(
                  '${(top.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: accentColor,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        Text('All scores', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        ...results.map(
          (r) => Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(child: Text(r.label, overflow: TextOverflow.ellipsis)),
                    Text('${(r.confidence * 100).toStringAsFixed(1)}%'),
                  ],
                ),
                const SizedBox(height: 2),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: r.confidence.clamp(0.0, 1.0),
                    minHeight: 6,
                    backgroundColor: Colors.black.withValues(alpha: 0.06),
                    color: accentColor,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _ModelLoadErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ModelLoadErrorView({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 12),
            const Text(
              'Model failed to load',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}
