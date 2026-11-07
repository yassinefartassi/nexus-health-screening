import 'package:flutter/material.dart';

import '../services/ocular_inference.dart';
import '../widgets/image_classification_view.dart';

class OcularScreen extends StatefulWidget {
  const OcularScreen({super.key});

  @override
  State<OcularScreen> createState() => _OcularScreenState();
}

class _OcularScreenState extends State<OcularScreen> {
  final _service = OcularInferenceService();

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ImageClassificationView(service: _service, accentColor: Colors.indigo);
  }
}
