{
  description = "flake template";

  inputs = {
    flake-lock.url = "github:wrvsrx/flake-lock";
    nixpkgs.follows = "flake-lock/nixpkgs";
    flake-parts.follows = "flake-lock/flake-parts";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { inputs, ... }:
      {
        systems = [ "x86_64-linux" ];
        perSystem =
          { pkgs, system, ... }:
          {
            _module.args.pkgs = import inputs.nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
            devShells.default = pkgs.callPackage ./shell.nix { };
            formatter = pkgs.nixfmt-rfc-style;
          };
      }
    );
}
