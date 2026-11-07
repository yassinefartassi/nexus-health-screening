/// One completed federated-learning round, exactly as recorded by the FL
/// server's real manifest (`nexus_pfe_server_fed/models/<module>/manifest.json`,
/// served read-only over HTTP by `fl_server/status_api.py`). Every field here
/// is a real number read off that JSON — nothing in this file is computed,
/// estimated, or simulated.
class FlRound {
  final int round;
  final String timestamp;
  final int numClients;
  final double? accuracy;
  final double? loss;
  final int commBytes;

  const FlRound({
    required this.round,
    required this.timestamp,
    required this.numClients,
    required this.accuracy,
    required this.loss,
    required this.commBytes,
  });

  factory FlRound.fromJson(Map<String, dynamic> json) {
    return FlRound(
      round: json['round'] as int,
      timestamp: json['timestamp'] as String? ?? '',
      numClients: json['num_clients'] as int? ?? 0,
      accuracy: (json['accuracy'] as num?)?.toDouble(),
      loss: (json['loss'] as num?)?.toDouble(),
      commBytes: json['comm_bytes'] as int? ?? 0,
    );
  }
}

/// Round history for one federated module (dermatology or ocular), as
/// reported by `fl_server/status_api.py`'s `GET /status/<module>`.
class ModuleFlStatus {
  final String module;
  final String framework;
  final List<String> classNames;
  final List<FlRound> rounds;

  const ModuleFlStatus({
    required this.module,
    required this.framework,
    required this.classNames,
    required this.rounds,
  });

  FlRound? get latestRound => rounds.isEmpty ? null : rounds.last;
  bool get hasRun => rounds.isNotEmpty;

  factory ModuleFlStatus.fromJson(String module, Map<String, dynamic> json) {
    final roundsJson = (json['rounds'] as List<dynamic>?) ?? const [];
    return ModuleFlStatus(
      module: module,
      framework: json['framework'] as String? ?? '?',
      classNames:
          (json['class_names'] as List<dynamic>? ?? const [])
              .map((e) => e.toString())
              .toList(),
      rounds: roundsJson
          .map((e) => FlRound.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}
