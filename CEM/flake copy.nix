{
  inputs = {
    mach-nix.url = "github:DavHau/mach-nix?tag=3.5.0";
  };

  # See: https://gist.github.com/piperswe/6be06f58ba3925801b0dcceab22c997b for more on nix multiarch images.

  outputs = {self, nixpkgs, mach-nix }@inp:
    let
      requirements = ''
        numpy ~= 1.23.3
      '';
      python = "python310";
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
        docker =
        let image = mach-nix.lib."${system}".mkDockerImage {
          inherit requirements;
          inherit python;
        };
        in
        image.override (oa:{
          name = "cem";
          tag = "main";
          contents = [ self ];
          config.Cmd = oa.config.Cmd ++ ["main.py"];
        });
      });
    };
}