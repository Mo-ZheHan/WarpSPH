{
  cmake,
  meson,
  pkg-config,
  ninja,
  poetry,
  cudaPackages_12_4,
  cudaPackages ? cudaPackages_12_4,
  llvmPackages_19,
  llvmPackages' ? llvmPackages_19,
  mkShell,
  lib,
  stdenv,
  zlib,
}:
mkShell {
  env.LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + lib.makeLibraryPath [
    stdenv.cc.cc
    zlib
  ];
  shellHook = ''
    source $(poetry env info -p)/bin/activate
  '';
  nativeBuildInputs = [
    cmake
    meson
    pkg-config
    ninja
    llvmPackages'.clang-tools
    poetry
  ];
  buildInputs = [
    cudaPackages.cuda_cccl
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cudart
    cudaPackages.libcurand
  ];
}
