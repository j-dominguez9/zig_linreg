# shell.nix
let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    packages = [
      pkgs.zig
      pkgs.zls
      # other deps here
    ];
  }
