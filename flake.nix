{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
    # Only the sim's Go toolchain comes from here. Its go.mod needs go 1.27, which 25.11 lacks.
    nixpkgs-go.url = "github:nixos/nixpkgs?ref=nixos-26.05";
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
          "aarch64-linux" # the e2e-tests CI job runs on GitHub's ARM runners
        ];
        flake = {
          lib = {
            pyproject = pyproject-nix.lib.project.loadPyproject {
              projectRoot = self;
            };
          };
        };
        perSystem =
          { pkgs, self', system, ... }@systemInputs:
          let
            python = pkgs.python3;
            goPkgs = inputs.nixpkgs-go.legacyPackages.${system};
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
                  prometheus
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
                export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
                python -m venv .venv
                source .venv/bin/activate
                pdm sync -d
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

              llm-d-inference-sim = (pkgs.buildGoModule.override { go = goPkgs.go_1_27; }) rec {
                pname = "llm-d-inference-sim";
                version = "1.0.0";

                src = pkgs.fetchFromGitHub {
                  owner = "llm-d";
                  repo = "llm-d-inference-sim";
                  tag = "v${version}";
                  hash = "sha256-GI1sF1+XgB7o+4VSRwNv0UBFJybzNX4cZSrPvNYLi1E=";
                };
                vendorHash = "sha256-pVacBlTPqOkLekVIOWmeONxVtobCHnUAp66ExBnn17g=";

                # cmd/dataset-tool is a second binary the e2e suite does not use.
                subPackages = [ "cmd/llm-d-inference-sim" ];

                # several tests require networking.
                doCheck = false;

                meta = {
                  description = "A light weight vLLM simulator, for mocking out replicas";
                  homepage = "https://github.com/llm-d/llm-d-inference-sim";
                  license = with nixpkgs.lib.licenses; asl20;
                  mainProgram = "llm-d-inference-sim";
                };
              };
            };
          };
      }
    );
}
