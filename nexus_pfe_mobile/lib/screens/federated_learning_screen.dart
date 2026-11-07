// "Apprentissage fédéré" screen: explains and shows how NEXUS_PFE's
// federated learning works, using REAL round history from the FL server
// (never fabricated numbers — see fl_status_service.dart) and an honest
// account of what actually trains where.
//
// ---------------------------------------------------------------------
// WHAT'S REAL VS SIMULATED ON THIS SCREEN (read this before changing it)
// ---------------------------------------------------------------------
// On-device training in Flutter/Dart is not implemented here, and this is a
// deliberate proof-of-concept scope decision, not a missing feature: none of
// Flutter's current ML stacks (TFLite, ONNX Runtime Mobile, ...) support
// running Flower's client-side training loop with backpropagation through
// EfficientNetV2-S / ResNet-18 the way TensorFlow/PyTorch do on desktop, and
// faking that with a progress bar that doesn't correspond to any real
// gradient step would misrepresent the thesis's actual FL results. So:
//
//   - REAL: the round history shown below (round number, client count,
//     held-out accuracy/loss) is fetched live from nexus_pfe_server_fed's
//     status API and reflects actual FedAvg rounds that actually ran.
//   - REAL: the federated clients that produce that history are
//     nexus_pfe's `fl_client/` (Python/desktop) and
//     nexus_pfe_raspberrypi's `fl/` — both train for real, in TensorFlow or
//     PyTorch, and send only weight updates to the server.
//   - SIMULATED / NOT IMPLEMENTED: this mobile app does not itself train a
//     model or send weight updates. The "participate" section below is a
//     trigger/handoff, not a fake local training run — it hands the user
//     the exact command to launch a real Python client (optionally on a
//     companion machine on the same LAN), it does not pretend to run one
//     itself.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;

import '../models/fl_status.dart';
import '../services/fl_status_service.dart';

class FederatedLearningScreen extends StatefulWidget {
  const FederatedLearningScreen({super.key});

  @override
  State<FederatedLearningScreen> createState() =>
      _FederatedLearningScreenState();
}

class _FederatedLearningScreenState extends State<FederatedLearningScreen> {
  final _urlController = TextEditingController(
    text: FlStatusService.defaultBaseUrl,
  );

  bool _loading = false;
  bool _attempted = false;
  Map<String, ModuleFlStatus>? _status;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _refresh() async {
    setState(() => _loading = true);
    final service = FlStatusService(baseUrl: _urlController.text.trim());
    final result = await service.fetchStatus();
    if (!mounted) return;
    setState(() {
      _status = result;
      _loading = false;
      _attempted = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Apprentissage fédéré')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _PrivacyBanner(),
            const SizedBox(height: 20),
            Text('Le cycle FedAvg', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            const _CycleDiagram(),
            const SizedBox(height: 8),
            const Text(
              'Dermatologie (EfficientNetV2-S, TensorFlow/Keras) et oculaire '
              '(ResNet-18, PyTorch) ont chacun leur propre modèle global '
              'versionné indépendamment — jamais de poids convertis entre les '
              'deux. rPPG (CHROM/POS) n\'a pas de poids entraînables et n\'est '
              'donc jamais fédéré.',
              style: TextStyle(color: Colors.black54, fontSize: 13),
            ),
            const SizedBox(height: 24),
            const _RealVsSimulatedCard(),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: Text(
                    'État réel des rounds exécutés',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                IconButton(
                  onPressed: _loading ? null : _refresh,
                  icon: _loading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.refresh),
                ),
              ],
            ),
            const SizedBox(height: 4),
            TextField(
              controller: _urlController,
              decoration: const InputDecoration(
                labelText: 'Adresse de l\'API de statut du serveur FL',
                helperText:
                    'nexus_pfe_server_fed, scripts\\start_status_api.ps1 '
                    '(défaut : port 8090)',
                border: OutlineInputBorder(),
                isDense: true,
              ),
              onSubmitted: (_) => _refresh(),
            ),
            const SizedBox(height: 16),
            _buildStatusBody(),
            const SizedBox(height: 24),
            const _ParticipateCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusBody() {
    if (!_attempted && _loading) {
      return const Center(child: Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: CircularProgressIndicator(),
      ));
    }
    if (_status == null) {
      return _WaitingCard(
        message:
            'Impossible de joindre l\'API de statut du serveur FL à cette '
            'adresse. Démarrez le serveur FL et son API de statut, ou '
            'vérifiez l\'adresse ci-dessus. Aucune métrique n\'est affichée '
            'tant qu\'un round réel n\'a pas eu lieu.',
      );
    }
    final derm = _status!['dermatology'];
    final ocular = _status!['ocular'];
    return Column(
      children: [
        _ModuleStatusCard(label: 'Dermatologie', status: derm),
        const SizedBox(height: 12),
        _ModuleStatusCard(label: 'Oculaire', status: ocular),
      ],
    );
  }
}

class _PrivacyBanner extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.green.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.green.withValues(alpha: 0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.lock_outline, color: Colors.green),
          const SizedBox(width: 10),
          Expanded(
            child: RichText(
              text: const TextSpan(
                style: TextStyle(color: Colors.black87, fontSize: 13, height: 1.4),
                children: [
                  TextSpan(
                    text: 'Vos images ne quittent jamais l\'appareil. ',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  TextSpan(
                    text:
                        'Seuls les poids du modèle (des tableaux de nombres '
                        'décrivant ce que le réseau a appris) sont envoyés au '
                        'serveur d\'agrégation — jamais une photo, jamais une '
                        'donnée patient.',
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CycleStep {
  final IconData icon;
  final String label;
  final String desc;
  const _CycleStep(this.icon, this.label, this.desc);
}

const _cycleSteps = [
  _CycleStep(Icons.phone_android, 'Client', 'Entraîne localement, données jamais partagées'),
  _CycleStep(Icons.arrow_forward, 'Poids', 'Seuls les poids locaux sont envoyés'),
  _CycleStep(Icons.dns_outlined, 'Serveur', 'Reçoit les poids de chaque client'),
  _CycleStep(Icons.functions, 'FedAvg', 'Moyenne pondérée des poids reçus'),
  _CycleStep(Icons.public, 'Modèle global', 'Nouvelle version : global_v{round}'),
  _CycleStep(Icons.replay, 'Redistribution', 'Sert de départ au round suivant'),
];

class _CycleDiagram extends StatelessWidget {
  const _CycleDiagram();

  @override
  Widget build(BuildContext context) {
    // Fixed height sized generously above the tallest card's actual content
    // (icon + 1-line label + up to 3-line description at fontSize 10) so
    // nothing clips under normal text scale. Found overflowing on a real
    // device at the previous height (132) -- the description Text had no
    // maxLines/Expanded, so a 3-line wrap (e.g. "Modèle global"'s desc) blew
    // past the available height. Both the explicit maxLines/ellipsis below
    // and this larger height matter: maxLines alone still doesn't help if
    // the outer SizedBox is shorter than 3 lines' worth of text.
    return SizedBox(
      height: 152,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _cycleSteps.length,
        separatorBuilder: (_, _) => const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4),
          child: Icon(Icons.chevron_right, color: Colors.black26),
        ),
        itemBuilder: (context, i) {
          final step = _cycleSteps[i];
          return SizedBox(
            width: 116,
            child: Card(
              elevation: 0,
              color: Colors.indigo.withValues(alpha: 0.06),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(step.icon, color: Colors.indigo),
                    const SizedBox(height: 6),
                    Text(
                      step.label,
                      textAlign: TextAlign.center,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 12),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      step.desc,
                      textAlign: TextAlign.center,
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(color: Colors.black54, fontSize: 10),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _RealVsSimulatedCard extends StatelessWidget {
  const _RealVsSimulatedCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.amber.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.amber.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          Row(
            children: [
              Icon(Icons.info_outline, color: Colors.amber, size: 20),
              SizedBox(width: 8),
              Text('Réel vs simulé, sur ce téléphone', style: TextStyle(fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 8),
          Text(
            '• RÉEL : l\'historique des rounds ci-dessous vient du vrai serveur FL '
            '(FedAvg exécuté réellement).\n'
            '• RÉEL : l\'entraînement fédéré s\'exécute réellement, mais côté Python '
            '(client web nexus_pfe/fl_client, client Raspberry Pi nexus_pfe_raspberrypi/fl) '
            '— en TensorFlow/PyTorch.\n'
            '• SIMULÉ / NON IMPLÉMENTÉ : cette application mobile n\'entraîne PAS de '
            'modèle sur l\'appareil et n\'envoie aucun poids elle-même. Entraîner un '
            'réseau profond en Flutter/Dart n\'est pas réaliste avec les frameworks '
            'actuels — plutôt que de simuler un faux entraînement, cet écran affiche '
            'l\'état réel et permet de déclencher/documenter la participation via un '
            'client Python.',
            style: TextStyle(fontSize: 13, height: 1.4),
          ),
        ],
      ),
    );
  }
}

class _WaitingCard extends StatelessWidget {
  final String message;
  const _WaitingCard({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.blueGrey.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.hourglass_empty, color: Colors.blueGrey),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              '⏳ En attente d\'exécution. $message',
              style: const TextStyle(fontSize: 13, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}

class _ModuleStatusCard extends StatelessWidget {
  final String label;
  final ModuleFlStatus? status;

  const _ModuleStatusCard({required this.label, required this.status});

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: Theme.of(context).textTheme.titleSmall),
            if (status != null)
              Text(status!.framework, style: const TextStyle(color: Colors.black45, fontSize: 11)),
            const SizedBox(height: 10),
            if (status == null || !status!.hasRun)
              const Text(
                '⏳ En attente d\'exécution — aucun round FL n\'a encore été '
                'effectué pour ce module.',
                style: TextStyle(color: Colors.black54, fontSize: 13),
              )
            else
              _ModuleStatusBody(status: status!),
          ],
        ),
      ),
    );
  }
}

class _ModuleStatusBody extends StatelessWidget {
  final ModuleFlStatus status;
  const _ModuleStatusBody({required this.status});

  @override
  Widget build(BuildContext context) {
    final latest = status.latestRound!;
    final accuracyPct = latest.accuracy != null ? '${(latest.accuracy! * 100).toStringAsFixed(1)}%' : '—';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            _StatChip(label: 'Round', value: '${latest.round}'),
            const SizedBox(width: 8),
            _StatChip(label: 'Clients', value: '${latest.numClients}'),
            const SizedBox(width: 8),
            _StatChip(label: 'Précision', value: accuracyPct),
          ],
        ),
        const SizedBox(height: 14),
        const Text('Précision par round (jeu de test réservé)', style: TextStyle(fontSize: 12, color: Colors.black54)),
        const SizedBox(height: 6),
        _AccuracyBars(rounds: status.rounds),
      ],
    );
  }
}

class _StatChip extends StatelessWidget {
  final String label;
  final String value;
  const _StatChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.indigo.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: const TextStyle(fontSize: 10, color: Colors.black54)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
        ],
      ),
    );
  }
}

/// Simple bar chart of accuracy-per-round built from plain widgets (no
/// charting package dependency) — every bar's height is a real accuracy
/// value from [ModuleFlStatus.rounds], never interpolated or invented.
class _AccuracyBars extends StatelessWidget {
  final List<FlRound> rounds;
  const _AccuracyBars({required this.rounds});

  @override
  Widget build(BuildContext context) {
    const maxHeight = 70.0;
    return SizedBox(
      height: maxHeight + 20,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: rounds.map((r) {
          final fraction = (r.accuracy ?? 0).clamp(0.0, 1.0);
          return Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 3),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Container(
                    height: maxHeight * fraction,
                    decoration: BoxDecoration(
                      color: Colors.indigo,
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(3)),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text('R${r.round}', style: const TextStyle(fontSize: 10, color: Colors.black54)),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

class _ParticipateCard extends StatelessWidget {
  const _ParticipateCard();

  static const _command =
      'python -m fl_client.client --client-id 0 --server <ip-serveur>:8080 '
      '--model dermatology --num-clients 2';

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Participer à un round', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            const Text(
              'L\'entraînement réel se fait via un client Python (voir ci-dessus), '
              'pas sur ce téléphone. Ce bouton copie la commande à lancer sur une '
              'machine Python (le client web nexus_pfe, ou tout ordinateur du '
              'même réseau local) pour participer réellement à un round — cet '
              'écran ne simule aucun entraînement local.',
              style: TextStyle(fontSize: 13, color: Colors.black54, height: 1.4),
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Text(
                _command,
                style: TextStyle(
                  color: Colors.greenAccent,
                  fontFamily: 'monospace',
                  fontSize: 11,
                ),
              ),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: () async {
                await Clipboard.setData(const ClipboardData(text: _command));
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Commande copiée')),
                  );
                }
              },
              icon: const Icon(Icons.copy_outlined, size: 18),
              label: const Text('Copier la commande'),
            ),
          ],
        ),
      ),
    );
  }
}
