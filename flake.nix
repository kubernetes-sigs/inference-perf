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

              llm-d-inference-sim =
                let
                  # create necessary python for kv-cache-manager-wrapper.
                  neededPython = pkgs.python312.withPackages (
                    ps: with ps; [
                      packaging
                      pillow
                      torch
                      transformers
                      jinja2
                    ]
                  );
                in
                pkgs.buildGoModule rec {
                  pname = "llm-d-inference-sim";
                  version = "0.7.1";

                  src = pkgs.fetchFromGitHub {
                    owner = "llm-d";
                    repo = "llm-d-inference-sim";
                    tag = "v${version}";
                    hash = "sha256-PFXqhA1Dz8xg2a7RtRcWE11RIzovaEduZT1G7oAUTS0=";
                  };
                  vendorHash = "sha256-8+W3FloObny7ZWq5h02yWF4skOE2gRbceCtWBzmZslE=";

                  nativeBuildInputs = with pkgs; [
                    pkg-config
                    makeWrapper
                    neededPython
                  ];

                  buildInputs = with pkgs; [
                    zeromq
                    libtokenizers
                    neededPython
                  ];

                  preBuild = ''
                    # https://github.com/llm-d/llm-d-inference-sim/blob/cf682b5a7b160e27754e9b186b7e2dfeb24678bb/Dockerfile#L52-L53
                    export CGO_CFLAGS="''${CGO_CFLAGS:-} $(${neededPython.executable}-config --cflags)"
                    export CGO_LDFLAGS="''${CGO_LDFLAGS:-} $(${neededPython.executable}-config --ldflags --embed)"
                  '';

                  postInstall = ''
                    mkdir -p $out/lib/python3.12/site-packages

                    cp \
                      vendor/github.com/llm-d/llm-d-kv-cache-manager/pkg/preprocessing/chat_completions/*.py \
                      $out/lib/python3.12/site-packages/

                    wrapProgram $out/bin/llm-d-inference-sim \
                      --prefix PYTHONPATH : $out/lib/python3.12/site-packages \
                      --set    PYTHON ${neededPython}
                  '';

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
