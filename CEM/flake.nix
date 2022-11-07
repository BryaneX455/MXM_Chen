{
  inputs = {
    mach-nix.url = "github:DavHau/mach-nix?tag=3.5.0";
  };

  # See: https://gist.github.com/piperswe/6be06f58ba3925801b0dcceab22c997b for more on nix multiarch images.

  outputs = {self, nixpkgs, mach-nix }@inp:
    let
      requirements = builtins.readFile ./requirements.txt;
      python = "python310";
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