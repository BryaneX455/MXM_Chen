{
  # nixConfig.extra-substituters = [
  #   "https://tweag-jupyter.cachix.org"
  # ];
  # nixConfig.extra-trusted-public-keys = [
  #   "tweag-jupyter.cachix.org-1:UtNH4Zs6hVUFpFBTLaA4ejYavPo5EFFqgd7G7FxGW9g="
  # ];

  inputs = {
    mach-nix.url = "github:DavHau/mach-nix";
    # jupyter.url = "github:tweag/jupyterWith";
  };

  # See: https://gist.github.com/piperswe/6be06f58ba3925801b0dcceab22c997b for more on nix multiarch images.

  outputs = {self, nixpkgs, mach-nix }@inp:
    let
      requirements = builtins.readFile ./requirements.txt;
      python = "python39";
      name = "cem";
      tag = "main";

      supportedSystems = [ "aarch64-linux" "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;}));
    in
    {
      packages = forAllSystems (system: pkgs: {
        default = mach-nix.lib."${system}".mkPython {
          inherit requirements;
          inherit python;
        };
      });

      # packages = forAllSystems (system: pkgs: let
      #   machNix = mach-nix.lib."${system}".mkPython {
      #     inherit requirements;
      #     inherit python;
      #   };
      #   iPython = jupyter.jupyterKernels.python {
      #     name = "mach-nix-jupyter";
      #     python3 = machNix.python;
      #     packages = machNix.python.pkgs.selectPkgs;
      #   };
      #   jupyterEnvironment = jupyter.lib.x86_64-linux.mkJupyterlab {
      #     kernels = (k: [ iPython ]);
      #   };
      #   in
      #   jupyterEnvironment.env
      # );

      apps = forAllSystems (system: pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default.out}/bin/python";
        };
      });

      docker = let
        # architectures = [ "i686" "x86_64" "aarch64" "powerpc64le" ];
        # crossSystems = map (arch: {
        #   inherit arch;
        #   pkgs = (import nixpkgs {
        #     crossSystem = { config = "${arch}-unknown-linux-musl"; };
        #   }).pkgsStatic;
        # }) architectures;
        # buildImage = arch: 
        image = mach-nix.lib."aarch64-linux".mkDockerImage {
          inherit requirements;
          inherit python;
        };
      in image.override (oa:{
        inherit name;
        inherit tag;
        contents = [ self ];
        config.Cmd = oa.config.Cmd ++ ["main.py"];
      });

    };
}