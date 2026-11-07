// HTTP client for nexus_pfe_server_fed's read-only FL status API
// (fl_server/status_api.py). This is the ONLY network call this app makes
// anywhere — see home_screen.dart's "no server calls" line, which stays true
// for every inference path; this fetches small JSON round summaries the
// server already wrote to disk, never an image, never a model weight.
//
// If the server/status API isn't reachable, callers get null back and the
// federated_learning_screen.dart must render "waiting for a run" — never a
// fabricated chart. See that server-side file's docstring for why this
// exists as a separate, read-only endpoint from the actual Flower FedAvg
// server (gRPC, a different port).
import 'dart:convert';
import 'dart:developer' as developer;

import 'package:http/http.dart' as http;

import '../models/fl_status.dart';

class FlStatusService {
  /// Default assumes the FL server's status API runs on the same machine
  /// this app's dev host reaches via `adb reverse` / emulator loopback, or
  /// the desktop during local testing. Real LAN use requires the server's
  /// actual IP — see the screen's editable address field and the project
  /// README's federated learning section.
  static const String defaultBaseUrl = 'http://127.0.0.1:8090';

  final String baseUrl;

  const FlStatusService({this.baseUrl = defaultBaseUrl});

  /// GET {baseUrl}/status -> one ModuleFlStatus per module, or null if the
  /// status API isn't reachable (server offline, wrong address, no run yet
  /// at all — an empty-but-reachable response still returns real, just-empty
  /// ModuleFlStatus objects, which is different from "unreachable").
  Future<Map<String, ModuleFlStatus>?> fetchStatus() async {
    try {
      final uri = Uri.parse('$baseUrl/status');
      final response = await http.get(uri).timeout(const Duration(seconds: 4));
      if (response.statusCode != 200) return null;

      final decoded = jsonDecode(response.body) as Map<String, dynamic>;
      return decoded.map(
        (module, value) => MapEntry(
          module,
          ModuleFlStatus.fromJson(module, value as Map<String, dynamic>),
        ),
      );
    } catch (e, st) {
      developer.log(
        'Failed to reach FL status API at $baseUrl',
        name: 'FlStatusService',
        error: e,
        stackTrace: st,
      );
      return null;
    }
  }
}
