# Required by flutter_onnxruntime: keeps the ONNX Runtime Android bindings
# from being stripped/renamed by R8 in release builds.
-keep class ai.onnxruntime.** { *; }
