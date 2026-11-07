import 'package:flutter/material.dart';

import 'dermatology_screen.dart';
import 'federated_learning_screen.dart';
import 'ocular_screen.dart';
import 'rppg_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('NEXUS PFE')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 8),
              child: Text(
                'All inference (Ocular, Dermatology, rPPG) runs fully '
                'on-device — no server calls, no internet required. The '
                '"Apprentissage fédéré" screen is the one exception: it '
                'polls the FL server\'s read-only status API to show real '
                'round history, never an image.',
                style: TextStyle(color: Colors.black54),
              ),
            ),
            const SizedBox(height: 8),
            _ModuleCard(
              icon: Icons.remove_red_eye_outlined,
              title: 'Ocular',
              subtitle: 'Eye disease classification from a photo',
              color: Colors.indigo,
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const OcularScreen()),
              ),
            ),
            const SizedBox(height: 12),
            _ModuleCard(
              icon: Icons.healing_outlined,
              title: 'Dermatology',
              subtitle: 'Skin condition classification from a photo',
              color: Colors.teal,
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const DermatologyScreen()),
              ),
            ),
            const SizedBox(height: 12),
            _ModuleCard(
              icon: Icons.favorite_outline,
              title: 'rPPG',
              subtitle: 'Contactless heart rate & HRV from the live camera',
              color: Colors.redAccent,
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const RppgScreen()),
              ),
            ),
            const SizedBox(height: 12),
            _ModuleCard(
              icon: Icons.hub_outlined,
              title: 'Apprentissage fédéré',
              subtitle: 'Comment le modèle global s\'améliore sans jamais voir vos images',
              color: Colors.deepPurple,
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const FederatedLearningScreen()),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModuleCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _ModuleCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              CircleAvatar(
                radius: 26,
                backgroundColor: color.withValues(alpha: 0.12),
                child: Icon(icon, color: color, size: 28),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 4),
                    Text(
                      subtitle,
                      style: const TextStyle(color: Colors.black54),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: Colors.black38),
            ],
          ),
        ),
      ),
    );
  }
}
