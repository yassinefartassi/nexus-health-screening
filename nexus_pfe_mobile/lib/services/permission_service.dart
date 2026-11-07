import 'package:permission_handler/permission_handler.dart';

/// Thin wrapper around permission_handler used by every screen that touches
/// the camera (Ocular/Dermatology capture, rPPG live preview).
class PermissionService {
  /// Requests camera permission if not already granted. Returns true if the
  /// caller may proceed to open the camera.
  static Future<bool> ensureCameraPermission() async {
    final status = await Permission.camera.status;
    if (status.isGranted) return true;

    final result = await Permission.camera.request();
    return result.isGranted;
  }

  /// True if the user denied permission with "don't ask again" (Android) or
  /// the equivalent permanent denial on iOS — the caller should send them to
  /// system settings instead of requesting again.
  static Future<bool> isPermanentlyDenied() async {
    return Permission.camera.status.then((s) => s.isPermanentlyDenied);
  }

  static Future<void> openSettings() => openAppSettings();
}
