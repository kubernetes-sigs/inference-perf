{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-parts,
      pyproject-nix,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { config, ... }:
      {
        systems = [
          "x86_64-linux"
        ];
        flake = {
          lib = {
            pyproject = pyproject-nix.lib.project.loadPyproject {
              projectRoot = self;
            };
          };
        };
        perSystem =
          { pkgs, self', ... }@systemInputs:
          let
            python = pkgs.python3;
          in
          {
            devShells.default = pkgs.mkShell {
              # PATH-only packages:
              packages =
                with pkgs;
                with python.pkgs;
                with self'.packages;
                [
                  llm-d-inference-sim
                  pdm
                  python

                  # choose either python-lsp-server or pyright:
                  basedpyright
                  # python-lsp-server
                  # pylsp-mypy
                ];

              buildInputs =
                with pkgs;
                with python.pkgs;
                [
                  numpy
                  torch
                ];

              shellHook = ''
                python -m venv .venv
                source .venv/bin/activate
                pdm sync -d --no-self
              '';
            };

            packages = rec {
              default = inference-perf;

              inference-perf =
                let
                  buildAttrs = self.lib.pyproject.renderers.buildPythonPackage {
                    inherit python;
                  };
                in
                python.pkgs.buildPythonPackage (buildAttrs // { });

              llm-d-inference-sim = pkgs.buildGoModule rec {
                pname = "llm-d-inference-sim";
                version = "0.6.1";

                src = pkgs.fetchFromGitHub {
                  owner = "llm-d";
                  repo = "llm-d-inference-sim";
                  tag = "v${version}";
                  hash = "sha256-KdA7dgdy1jGjRhrqXfkg4Z9V3SXPcKp1FnTtm+e5DSA=";
                };
                vendorHash = "sha256-MINH7J2ozTORFK/KgZvXBlwThYRISL1wlHebdZxvuvw=";

                nativeBuildInputs = with pkgs; [
                  pkg-config
                ];

                buildInputs = with pkgs; [
                  zeromq
                  libtokenizers
                ];

                # several tests require networking.
                doCheck = false;

                meta = {
                  description = "A light weight vLLM simulator, for mocking out replicas";
                  homepage = "https://github.com/llm-d/llm-d-inference-sim";
                  license = with nixpkgs.lib.licenses; asl20;
                  mainProgram = "llm-d-inference-sim";
                };
              };

              libtokenizers = pkgs.rustPlatform.buildRustPackage rec {
                pname = "libtokenizers";
                version = "1.22.1"; # keep same as llm-d-inference-sim's version

                src = pkgs.fetchFromGitHub {
                  owner = "daulet";
                  repo = "tokenizers";
                  tag = "v${version}";
                  hash = "sha256-unGAXpD4GHWVFcXAwd0zU/u30wzH909tDcRYRPsSKwQ=";
                };
                cargoHash = "sha256-rY3YAcCbbx5CY6qu44Qz6UQhJlWVxAWdTaUSagHDn2o=";

                meta = {
                  description = "Go bindings for Tiktoken & HuggingFace Tokenizer";
                  homepage = "https://github.com/daulet/tokenizers";
                  license = with nixpkgs.lib.licenses; mit;
                };
              };
            };
          };
      }
    );
}
