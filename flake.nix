{
  description = "TradePulse reproducible development environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [];
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311Full
            python312Full
            python313Full
            uv
            redis
            cmake
            pkg-config
            zlib
            openssl
          ];
          shellHook = ''
            export UV_PYTHON=python3.12
            export TRADEPULSE_DEV_SHELL=1
            echo "Activating TradePulse dev shell with uv + pip-tools pins"
            echo "Run 'uv pip sync requirements-dev.lock' to install development dependencies."
          '';
        };
      });
}
