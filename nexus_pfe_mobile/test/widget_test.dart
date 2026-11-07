// Smoke test: verifies the app boots to the home screen with its three
// module entry points. Deeper tests (ONNX inference, camera, rPPG signal
// processing) need real assets/hardware and are exercised manually — see
// README.md.
import 'package:flutter_test/flutter_test.dart';

import 'package:nexus_pfe_mobile/main.dart';

void main() {
  testWidgets('Home screen shows all three modules', (WidgetTester tester) async {
    await tester.pumpWidget(const NexusPfeApp());

    expect(find.text('Ocular'), findsOneWidget);
    expect(find.text('Dermatology'), findsOneWidget);
    expect(find.text('rPPG'), findsOneWidget);
  });
}
