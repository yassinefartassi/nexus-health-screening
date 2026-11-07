import 'package:flutter/material.dart';

import '../services/dermatology_inference.dart';
import '../widgets/image_classification_view.dart';

class DermatologyScreen extends StatefulWidget {
  const DermatologyScreen({super.key});

  @override
  State<DermatologyScreen> createState() => _DermatologyScreenState();
}

class _DermatologyScreenState extends State<DermatologyScreen> {
  final _service = DermatologyInferenceService();

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ImageClassificationView(service: _service, accentColor: Colors.teal);
  }
}
